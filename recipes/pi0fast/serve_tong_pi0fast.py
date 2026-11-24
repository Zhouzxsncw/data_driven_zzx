#!/usr/bin/env python3
"""OpenPI π0-FAST TongBot websocket 服务端。

协议对齐 PI05 / ACoT（msgpack_numpy），复用 OpenpiPi05Runtime（其
create_trained_policy 对模型类型透明，PI0-FAST 的 TokenizeFASTInputs /
ExtractFASTActions transforms 由 pi0_fast_tongbot TrainConfig 提供）：

  handshake: {model: pi0_fast, action_horizon, action_dim, rtc_capable: false, ckpt}
  reset:     {reset: True}  → 空操作
  infer:     {obs: {state, images, prompt}} 或裸观测 {state, images, prompt}
  reply:     {actions: (H, D), server_timing}

RTC 不适用（FAST 是离散自回归解码，无 flow-matching 前缀可 inpaint）；
请求带 rtc 字段会被安全忽略。
"""
from __future__ import annotations

import argparse
import asyncio
import http
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_RECIPES_DIR = _THIS_DIR.parent
_REPO_ROOT = _RECIPES_DIR.parent
_INFER_ROOT = Path(os.environ.get("INFER_ROOT", _REPO_ROOT / "infer")).expanduser()
if str(_INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_INFER_ROOT))
if str(_RECIPES_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPES_DIR))

from _server_logger import ServerRequestLogger  # noqa: E402
from infer.policies._lingbot_msgpack_numpy import Packer, unpackb  # noqa: E402
from infer.policies.openpi_pi05_runtime import OpenpiPi05Runtime  # noqa: E402

import numpy as np
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("pi0fast_serve")

DEFAULT_CKPT = (
    # 训练 run 根目录；policy_config 会自动解析到最新 step 的 pretrained_model
    "/home/xk/VLA/openpi/cigai-agilex-train-openpi/checkpoints/"
    "00036_pi0fast_bs16_lr2.5e-5_ah15_1gpu"
)
DEFAULT_PROMPT = (
    "With both hands, pick up the yellow banana and the yellow lemon and put them "
    "into the black box; then take the green vegetables and place them into the blue box."
)


def _parse_delta_dims(raw: str) -> list[int]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return [7, 7, -1, -1]
    return [int(p) for p in parts]


def _parse_infer_payload(request: Any) -> Tuple[str, Optional[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    """Return ('reset', None) or ('infer', (obs, extras))."""
    if not isinstance(request, dict):
        raise ValueError(f"期望 dict 请求，得到 {type(request).__name__}")
    if request.get("reset"):
        return "reset", None
    extras: Dict[str, Any] = {"prev_action_chunk": None, "executed_steps": 0}
    if "obs" in request:
        obs = request["obs"]
        if not isinstance(obs, dict):
            raise ValueError("obs 必须是 dict")
        extras["prev_action_chunk"] = request.get("prev_action_chunk")
        extras["executed_steps"] = int(request.get("executed_steps") or 0)
        return "infer", (obs, extras)
    if "state" in request and "images" in request:
        extras["prev_action_chunk"] = request.get("prev_action_chunk")
        extras["executed_steps"] = int(request.get("executed_steps") or 0)
        return "infer", (request, extras)
    raise ValueError("无法识别的推理请求；需要 obs={state,images,prompt} 或裸 {state,images,prompt}")


class Pi0FastPolicy:
    """Thin wrapper around OpenpiPi05Runtime for websocket handlers."""

    def __init__(self, runtime: OpenpiPi05Runtime) -> None:
        self.runtime = runtime

    def infer(self, obs: Dict[str, Any], extras: Dict[str, Any]) -> Dict[str, Any]:
        prev = extras.get("prev_action_chunk")
        if prev is not None:
            prev = np.asarray(prev, dtype=np.float32)
        # FAST 无 RTC：runtime.infer 的 rtc 参数恒传 None
        raw, actions = self.runtime.infer(
            obs,
            prev_action_chunk=None,  # FAST 无前缀动作概念
            executed_steps=0,
        )
        return {
            "actions": np.asarray(actions, dtype=np.float32),
            "_timing": {},
            "_raw_action_range": (
                float(np.min(raw)) if raw.size else 0.0,
                float(np.max(raw)) if raw.size else 0.0,
            ),
        }


class WebsocketPolicyServer:
    def __init__(
        self,
        policy: Pi0FastPolicy,
        host: str,
        port: int,
        metadata: dict | None = None,
        srv_logger: ServerRequestLogger | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._srv_logger = srv_logger
        self._request_id = 0

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        if self._srv_logger is not None:
            self._srv_logger.start()
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_interval=None,
            ping_timeout=None,
        ) as server:
            LOGGER.info("listening ws://%s:%s", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        client_addr = str(websocket.remote_address)
        LOGGER.info("client %s connected", client_addr)
        if self._srv_logger:
            self._srv_logger.log_connection(
                client_addr=client_addr, event="connect", metadata=self._metadata
            )
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            t0 = time.monotonic()
            try:
                request = unpackb(await websocket.recv())
                self._request_id += 1
                kind, payload = _parse_infer_payload(request)
                if kind == "reset":
                    result = {"reset": True}
                    infer_ms = 0.0
                else:
                    obs, extras = payload  # type: ignore[misc]
                    result = self._policy.infer(obs, extras)
                    infer_ms = (time.monotonic() - t0) * 1000
                    LOGGER.info(
                        "infer #%d actions=%s infer_ms=%.1f",
                        self._request_id,
                        tuple(np.asarray(result["actions"]).shape),
                        infer_ms,
                    )

                result.pop("_timing", None)
                result.pop("_raw_action_range", None)
                if not isinstance(result, dict):
                    result = {"actions": result}
                result["server_timing"] = {"infer_ms": infer_ms}

                t_send = time.monotonic()
                await websocket.send(packer.pack(result))
                send_ms = (time.monotonic() - t_send) * 1000

                if self._srv_logger:
                    self._srv_logger.log_request(
                        request_id=self._request_id,
                        client_addr=client_addr,
                        infer_ms=infer_ms,
                        request_data=request if isinstance(request, dict) else None,
                        response_data=result,
                        timing={"total_ms": round(infer_ms, 2), "network_send_ms": round(send_ms, 2)},
                    )
            except websockets.ConnectionClosed:
                LOGGER.info("client %s closed", client_addr)
                if self._srv_logger:
                    self._srv_logger.log_connection(client_addr=client_addr, event="disconnect")
                break
            except Exception:
                tb = traceback.format_exc()
                LOGGER.exception("infer failed")
                if self._srv_logger:
                    self._srv_logger.log_request(
                        request_id=self._request_id,
                        client_addr=client_addr,
                        infer_ms=(time.monotonic() - t0) * 1000,
                        error=tb.splitlines()[-1] if tb else "unknown",
                    )
                await websocket.send(tb)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error",
                )
                break


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _build_runtime(args: argparse.Namespace) -> OpenpiPi05Runtime:
    cameras = [c.strip() for p in args.cameras.split(",") if (c := p.strip())]
    runtime = OpenpiPi05Runtime(
        ckpt_dir=Path(args.ckpt).expanduser(),
        config_name=str(args.config_name),
        repo_path=Path(args.repo).expanduser() if args.repo else None,
        camera_names=cameras or ["left", "front", "right"],
        action_dim=int(args.action_dim),
        action_horizon=int(args.action_horizon),
        image_size=int(args.image_size),
        default_prompt=str(args.instruction),
        adapt_to_pi=bool(args.adapt_to_pi),
        delta_action_dims=_parse_delta_dims(args.delta_action_dims),
    )
    runtime.load()
    return runtime


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.environ.get("PI0FAST_CKPT", DEFAULT_CKPT))
    p.add_argument("--host", default=os.environ.get("PI0FAST_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PI0FAST_PORT", "1243")))
    p.add_argument("--instruction", default=os.environ.get("PI0FAST_PROMPT", DEFAULT_PROMPT))
    p.add_argument(
        "--config-name", default=os.environ.get("PI0FAST_CONFIG_NAME", "pi0_fast_tongbot")
    )
    p.add_argument("--cameras", default=os.environ.get("PI0FAST_CAMERAS", "left,front,right"))
    p.add_argument("--action-dim", type=int, default=int(os.environ.get("PI0FAST_ACTION_DIM", "16")))
    p.add_argument(
        "--action-horizon",
        type=int,
        default=int(os.environ.get("PI0FAST_ACTION_HORIZON", "15")),
    )
    p.add_argument("--image-size", type=int, default=int(os.environ.get("PI0FAST_IMAGE_SIZE", "224")))
    p.add_argument(
        "--delta-action-dims", default=os.environ.get("PI0FAST_DELTA_ACTION_DIMS", "7,7,-1,-1")
    )
    p.add_argument("--adapt-to-pi", action="store_true")
    p.add_argument("--repo", default=os.environ.get("PI0FAST_OPENPI_REPO", ""))
    p.add_argument("--load-only", action="store_true")
    p.add_argument(
        "--log-dir",
        default=os.environ.get("PI0FAST_LOG_DIR", ""),
        help="请求级 JSONL 日志目录（默认与 serve 脚本同目录）",
    )
    p.add_argument("--no-request-log", action="store_true")
    args = p.parse_args()

    ckpt = Path(args.ckpt).expanduser()
    # 接受: params/ 或 model.safetensors 直接存在，或训练 run 根目录
    # （checkpoints/last -> {step}；runtime 的 norm_stats 解析要求 ckpt
    # 指向 <step>/pretrained_model，故在此提前展开）
    if (ckpt / "checkpoints" / "last").exists():
        step_dir = (ckpt / "checkpoints" / "last").resolve()
        pretrained = step_dir / "pretrained_model"
        if pretrained.is_dir():
            ckpt = pretrained
            args.ckpt = str(ckpt)
            LOGGER.info("解析训练 run 最新 ckpt: %s", ckpt)
    if (
        not (ckpt / "model.safetensors").is_file()
        and not (ckpt / "params").exists()
        and not (ckpt / "checkpoints").is_dir()
    ):
        raise SystemExit(
            f"权重未就绪: {ckpt}（需要 model.safetensors 或 params/；"
            "训练 run 根目录也可，会自动解析最新 step）"
        )

    load_t0 = time.time()
    runtime = _build_runtime(args)
    load_sec = time.time() - load_t0
    LOGGER.info(
        "model ready action_horizon=%d action_dim=%d (%.1fs)",
        runtime.chunk_size,
        runtime.action_dim,
        load_sec,
    )

    srv_logger = None
    if not args.no_request_log:
        log_dir = Path(args.log_dir) if args.log_dir else _THIS_DIR
        log_path = log_dir / "pi0fast_requests.jsonl"
        srv_logger = ServerRequestLogger(log_path, gpu_device=0, model_name="pi0_fast")
        LOGGER.info("请求级日志: %s", log_path)

    if args.load_only:
        LOGGER.info("load-only OK")
        if srv_logger:
            srv_logger.close()
        return

    metadata = {
        "model": "pi0_fast",
        "ckpt": str(ckpt.resolve()),
        "action_horizon": int(runtime.chunk_size),
        "action_dim": int(runtime.action_dim),
        "rtc_capable": False,
        "config_name": str(args.config_name),
    }

    if srv_logger:
        srv_logger.log_model_loaded(
            model_name="pi0_fast",
            ckpt_path=str(ckpt.resolve()),
            load_time_sec=load_sec,
            extra={
                "action_horizon": int(runtime.chunk_size),
                "action_dim": int(runtime.action_dim),
            },
        )

    server = WebsocketPolicyServer(
        Pi0FastPolicy(runtime),
        host=str(args.host),
        port=int(args.port),
        metadata=metadata,
        srv_logger=srv_logger,
    )
    LOGGER.info("OpenPI π0-FAST 服务端就绪，等客户端连 ws://%s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
