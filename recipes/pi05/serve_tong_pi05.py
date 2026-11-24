#!/usr/bin/env python3
"""OpenPI π₀.₅ websocket 服务端（TongBot 真机 / Aloha / LIBERO sim）。

协议对齐 ACoT / OpenPI（msgpack_numpy），不要用 FastWAM 的 ``obs: [dict]``：

  handshake: {model: pi05, embodiment, action_horizon, action_dim, state_dim, rtc_capable, ckpt}
  reset:     {reset: True}  → 空操作（prev_chunk 由客户端跟踪）
  infer:     {obs: {state, images, prompt}, prev_action_chunk?, executed_steps?,
              rtc?, inference_delay?}
             或裸观测 {state, images, prompt}
             或 LIBERO {observation/image, observation/wrist_image, observation/state, prompt}
  reply:     {actions: (H, D), rtc_used, server_timing}

``--config`` 读同一份 YAML；``openpi_embodiment`` / ``PI05_EMBODIMENT`` 切换
TongBot 真机与 LIBERO sim。维数 / 相机由 profile 自动填。端口默认 1242。
RTC prefix inpainting 必须在本进程（JAX）里做；客户端只负责发上一包与已执行步数。
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
from typing import Any, Dict, Mapping, Optional, Tuple

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
from infer.policies.kemo_memory import KemoMemory  # noqa: E402
from infer.policies.pi05_embodiment import (  # noqa: E402
    DEFAULT_PI05_YAML,
    canonicalize_pi05_obs,
    get_pi05_embodiment,
    load_pi05_yaml,
)

import numpy as np
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("pi05_serve")

DEFAULT_CKPT = (
    "/home/xk/VLA/h100_ckpts/00092_pi05_jiangsuan1/012570/pretrained_model"
)
DEFAULT_PROMPT = (
    "With both hands cooperating, put the white medicine box into the green "
    "drawer with the blue handle, put the yellow medicine box into the blue "
    "drawer with the black handle, and then close the drawers."
)


def _join_cameras(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return ",".join(parts) if parts else None
    return str(value)


def _format_delta_dims(value: Any) -> Optional[str]:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(x)) for x in value)
    return str(value)


def _apply_yaml_to_args(args: argparse.Namespace, raw: Mapping[str, Any]) -> None:
    """Fill serve CLI defaults from the unified π₀.₅ YAML (already embodiment-merged)."""
    pol = raw.get("policy") if isinstance(raw.get("policy"), dict) else {}
    if not getattr(args, "embodiment", None):
        args.embodiment = raw.get("embodiment") or pol.get("openpi_embodiment")
    if not getattr(args, "config_name", None):
        args.config_name = pol.get("openpi_config_name")
    if not getattr(args, "ckpt", None) and pol.get("ckpt_dir"):
        args.ckpt = pol["ckpt_dir"]
    if not getattr(args, "instruction", None) and pol.get("task_prompt"):
        args.instruction = pol["task_prompt"]
    if not getattr(args, "cameras", None):
        joined = _join_cameras(pol.get("openpi_camera_names"))
        if joined:
            args.cameras = joined
    if getattr(args, "action_dim", None) is None and pol.get("openpi_action_dim") is not None:
        args.action_dim = int(pol["openpi_action_dim"])
    if getattr(args, "state_dim", None) is None and pol.get("openpi_state_dim") is not None:
        args.state_dim = int(pol["openpi_state_dim"])
    if getattr(args, "action_horizon", None) is None and pol.get("openpi_action_horizon") is not None:
        args.action_horizon = int(pol["openpi_action_horizon"])
    if getattr(args, "image_size", None) is None and pol.get("openpi_image_size") is not None:
        args.image_size = int(pol["openpi_image_size"])
    if getattr(args, "port", None) is None and pol.get("lingbot_port") is not None:
        args.port = int(pol["lingbot_port"])
    if getattr(args, "delta_action_dims", None) is None and "openpi_delta_action_dims" in pol:
        args.delta_action_dims = _format_delta_dims(pol.get("openpi_delta_action_dims"))
    if not getattr(args, "interleave_grippers", False) and pol.get("openpi_interleave_grippers"):
        args.interleave_grippers = True
    if getattr(args, "adapt_to_pi", None) is None and pol.get("openpi_adapt_to_pi") is not None:
        args.adapt_to_pi = bool(pol["openpi_adapt_to_pi"])
    if getattr(args, "vector_norm", None) in (None, "") and pol.get("openpi_vector_norm") is not None:
        args.vector_norm = pol.get("openpi_vector_norm")
    if (
        getattr(args, "num_inference_steps", None) is None
        and pol.get("num_inference_steps_override") is not None
    ):
        args.num_inference_steps = int(pol["num_inference_steps_override"])


def _load_serve_yaml(path: Optional[str], embodiment: Optional[str]) -> Optional[dict[str, Any]]:
    cfg_path = path
    if not cfg_path and DEFAULT_PI05_YAML.is_file():
        cfg_path = str(DEFAULT_PI05_YAML)
    if not cfg_path:
        return None
    resolved = Path(cfg_path).expanduser()
    if not resolved.is_file():
        raise SystemExit(f"找不到 π₀.₅ YAML: {resolved}")
    return load_pi05_yaml(resolved, embodiment=embodiment)


def _parse_delta_dims(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"none", "null", "-"}:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return default
    return str(val)


def _env_int(name: str) -> Optional[int]:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return None
    return int(val)


def _fill_infer_extras(request: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prev_action_chunk": request.get("prev_action_chunk"),
        "executed_steps": int(request.get("executed_steps") or 0),
        "rtc": request.get("rtc"),
        "inference_delay": int(request.get("inference_delay") or 0),
        "debug_attention": bool(request.get("debug_attention")),
    }


def _parse_infer_payload(
    request: Any,
) -> Tuple[str, Optional[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    """Return ('reset', None) or ('infer', (obs, extras))."""
    if not isinstance(request, dict):
        raise ValueError(f"期望 dict 请求，得到 {type(request).__name__}")
    if request.get("reset"):
        return "reset", None
    extras = _fill_infer_extras(request)
    if "obs" in request:
        obs = request["obs"]
        if not isinstance(obs, dict):
            raise ValueError("obs 必须是 dict")
        return "infer", (canonicalize_pi05_obs(obs), extras)
    if (
        ("state" in request and "images" in request)
        or "observation/image" in request
        or "observation/wrist_image" in request
    ):
        return "infer", (canonicalize_pi05_obs(request), extras)
    raise ValueError(
        "无法识别的推理请求；需要 obs={state,images,prompt}、裸 {state,images,prompt} "
        "或 LIBERO {observation/image, observation/wrist_image, observation/state}"
    )


class Pi05Policy:
    """Thin wrapper around OpenpiPi05Runtime for websocket handlers."""

    def __init__(self, runtime: OpenpiPi05Runtime, kemo_memory: KemoMemory | None = None) -> None:
        self.runtime = runtime
        self.kemo_memory = kemo_memory

    def reset(self) -> None:
        if self.kemo_memory is not None:
            self.kemo_memory.reset()
        self.runtime.reset_runtime()

    def infer(self, obs: Dict[str, Any], extras: Dict[str, Any]) -> Dict[str, Any]:
        if self.kemo_memory is not None:
            obs = self.kemo_memory.append(obs)
        prev = extras.get("prev_action_chunk")
        if prev is not None:
            # msgpack frombuffer 解出的数组只读，delta transform 需要原地写 → 复制可写副本
            prev = np.array(prev, dtype=np.float32, copy=True)
        rtc = extras.get("rtc")
        if rtc is not None and not isinstance(rtc, dict):
            rtc = None
        raw, actions = self.runtime.infer(
            obs,
            prev_action_chunk=prev,
            executed_steps=int(extras.get("executed_steps") or 0),
            rtc=rtc,
            inference_delay=int(extras.get("inference_delay") or 0),
            debug_attention=bool(extras.get("debug_attention")),
        )
        used_rtc = bool(getattr(self.runtime, "_last_rtc_used", False))
        rtc_info = dict(getattr(self.runtime, "_last_rtc", None) or {"used": used_rtc})
        reply = {
            "actions": np.asarray(actions, dtype=np.float32),
            "rtc_used": used_rtc,
            "rtc": rtc_info,
            "_timing": {},
            "_raw_action_range": (
                float(np.min(raw)) if raw.size else 0.0,
                float(np.max(raw)) if raw.size else 0.0,
            ),
        }
        attn = getattr(self.runtime, "_last_attention", None)
        if isinstance(attn, dict):
            reply["attention"] = attn
        fwd = getattr(self.runtime, "_last_forward", None)
        if isinstance(fwd, dict):
            reply["forward"] = fwd
        return reply


class WebsocketPolicyServer:
    def __init__(
        self,
        policy: Pi05Policy,
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
                        "infer #%d rtc_used=%s actions=%s infer_ms=%.1f",
                        self._request_id,
                        result.get("rtc_used"),
                        tuple(np.asarray(result["actions"]).shape),
                        infer_ms,
                    )

                timing = result.pop("_timing", None) if isinstance(result, dict) else None
                result.pop("_raw_action_range", None)
                if not isinstance(result, dict):
                    result = {"actions": result}
                result["server_timing"] = {"infer_ms": infer_ms}

                t_send = time.monotonic()
                await websocket.send(packer.pack(result))
                send_ms = (time.monotonic() - t_send) * 1000

                if self._srv_logger:
                    log_timing = dict(timing) if timing else {}
                    log_timing["total_ms"] = round(infer_ms, 2)
                    log_timing["network_send_ms"] = round(send_ms, 2)
                    extra = None
                    if isinstance(result, dict) and "rtc_used" in result:
                        extra = {
                            "rtc_used": bool(result["rtc_used"]),
                            "rtc": result.get("rtc"),
                        }
                    self._srv_logger.log_request(
                        request_id=self._request_id,
                        client_addr=client_addr,
                        infer_ms=infer_ms,
                        request_data=request if isinstance(request, dict) else None,
                        response_data=result,
                        timing=log_timing,
                        extra=extra,
                    )
            except websockets.ConnectionClosed:
                LOGGER.info("client %s closed", client_addr)
                if self._srv_logger:
                    self._srv_logger.log_connection(
                        client_addr=client_addr, event="disconnect"
                    )
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
    profile = get_pi05_embodiment(args.embodiment, args.config_name)
    config_name = str(args.config_name or profile.config_name)
    cameras_raw = str(args.cameras or "").strip()
    cameras = [c.strip() for p in cameras_raw.split(",") if (c := p.strip())]
    if not cameras:
        cameras = list(profile.cameras)
    action_dim = int(args.action_dim if args.action_dim is not None else profile.action_dim)
    state_dim = int(args.state_dim if args.state_dim is not None else profile.state_dim)
    action_horizon = int(
        args.action_horizon if args.action_horizon is not None else profile.action_horizon
    )
    if args.delta_action_dims is None and profile.delta_action_dims is not None:
        delta_dims = list(profile.delta_action_dims)
    else:
        delta_dims = _parse_delta_dims(args.delta_action_dims)
    if args.adapt_to_pi is None:
        adapt_to_pi = profile.adapt_to_pi
    else:
        adapt_to_pi = bool(args.adapt_to_pi)
    runtime = OpenpiPi05Runtime(
        ckpt_dir=Path(args.ckpt).expanduser(),
        config_name=config_name,
        repo_path=Path(args.repo).expanduser() if args.repo else None,
        camera_names=cameras,
        action_dim=action_dim,
        state_dim=state_dim,
        action_horizon=action_horizon,
        image_size=int(args.image_size),
        default_prompt=str(args.instruction),
        adapt_to_pi=adapt_to_pi,
        delta_action_dims=delta_dims,
        interleave_grippers=bool(getattr(args, "interleave_grippers", False)),
        vector_norm=getattr(args, "vector_norm", None),
        embodiment=profile.name,
        num_inference_steps=getattr(args, "num_inference_steps", None),
    )
    runtime.load()
    return runtime


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=_env_str("PI05_CONFIG"),
        help="统一 YAML（tongbot/libero 同一份）；默认 pi05_openpi_tongbot_16d_00092_ws_client.yaml",
    )
    p.add_argument("--ckpt", default=_env_str("PI05_CKPT"))
    p.add_argument("--host", default=_env_str("PI05_HOST") or "0.0.0.0")
    p.add_argument("--port", type=int, default=_env_int("PI05_PORT"))
    p.add_argument("--instruction", default=_env_str("PI05_PROMPT"))
    p.add_argument(
        "--embodiment",
        default=_env_str("PI05_EMBODIMENT"),
        help="tongbot | aloha | libero | robocasa365 | robotwin（默认读 YAML / PI05_EMBODIMENT）",
    )
    p.add_argument("--config-name", default=_env_str("PI05_CONFIG_NAME"))
    p.add_argument("--cameras", default=_env_str("PI05_CAMERAS"))
    p.add_argument("--action-dim", type=int, default=_env_int("PI05_ACTION_DIM"))
    p.add_argument("--state-dim", type=int, default=_env_int("PI05_STATE_DIM"))
    p.add_argument(
        "--action-horizon",
        type=int,
        default=_env_int("PI05_ACTION_HORIZON"),
    )
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=_env_int("PI05_NUM_INFERENCE_STEPS"),
        help="flow-matching denoise 步数（默认沿用 openpi 10）；等价于 YAML policy.num_inference_steps_override。",
    )
    p.add_argument("--image-size", type=int, default=_env_int("PI05_IMAGE_SIZE"))
    p.add_argument(
        "--delta-action-dims",
        default=_env_str("PI05_DELTA_ACTION_DIMS"),
        help="覆盖 TrainConfig delta mask；libero 默认不覆盖。空/none 表示不覆盖。",
    )
    p.add_argument(
        "--interleave-grippers",
        action="store_true",
        help="落盘 [左7|右7|左爪|右爪] 重排成 [左7|左爪|右7|右爪] 再做 delta。YAML 也可开。",
    )
    p.add_argument(
        "--vector-norm",
        default=_env_str("PI05_VECTOR_NORM"),
        help="state 输入与 action 输出共用：quantile 或 zscore。默认读 YAML openpi_vector_norm。",
    )
    p.add_argument("--adapt-to-pi", action="store_true")
    p.add_argument("--no-adapt-to-pi", action="store_true")
    p.add_argument("--repo", default=os.environ.get("PI05_OPENPI_REPO", ""))
    p.add_argument("--load-only", action="store_true")
    p.add_argument(
        "--log-dir",
        default=os.environ.get("PI05_LOG_DIR", ""),
        help="请求级 JSONL 日志目录（默认与 serve 脚本同目录）",
    )
    p.add_argument("--no-request-log", action="store_true")
    p.add_argument(
        "--kemo-memory-length",
        type=int,
        default=0,
        help="启用 KEMO observation memory；0 表示普通 pi0.5。训练约定建议使用 8。",
    )
    args = p.parse_args()
    if args.no_adapt_to_pi and args.adapt_to_pi:
        raise SystemExit("不能同时指定 --adapt-to-pi 与 --no-adapt-to-pi")
    if args.no_adapt_to_pi:
        args.adapt_to_pi = False
    elif args.adapt_to_pi:
        args.adapt_to_pi = True
    else:
        args.adapt_to_pi = None

    yaml_cfg = _load_serve_yaml(args.config, args.embodiment)
    if yaml_cfg is not None:
        _apply_yaml_to_args(args, yaml_cfg)

    profile = get_pi05_embodiment(args.embodiment, args.config_name)
    if not args.config_name:
        args.config_name = profile.config_name
    args.embodiment = profile.name
    if not args.ckpt:
        args.ckpt = DEFAULT_CKPT
    if not args.instruction:
        args.instruction = DEFAULT_PROMPT
    if args.port is None:
        args.port = 1242
    if args.image_size is None:
        args.image_size = 224

    ckpt = Path(args.ckpt).expanduser()
    if not (ckpt / "model.safetensors").is_file() and not (ckpt / "params").exists():
        raise SystemExit(f"权重未就绪: {ckpt}（需要 model.safetensors 或 params/）")

    load_t0 = time.time()
    runtime = _build_runtime(args)
    load_sec = time.time() - load_t0
    LOGGER.info(
        "model ready embodiment=%s action_horizon=%d state_dim=%d action_dim=%d (%.1fs)",
        runtime.embodiment,
        runtime.chunk_size,
        runtime.state_dim,
        runtime.action_dim,
        load_sec,
    )

    srv_logger = None
    if not args.no_request_log:
        log_dir = Path(args.log_dir) if args.log_dir else _THIS_DIR
        log_path = log_dir / "pi05_requests.jsonl"
        srv_logger = ServerRequestLogger(log_path, gpu_device=0, model_name="pi05")
        LOGGER.info("请求级日志: %s", log_path)

    if args.load_only:
        LOGGER.info("load-only OK")
        if srv_logger:
            srv_logger.close()
        return

    metadata = {
        "model": "pi05",
        "ckpt": str(ckpt.resolve()),
        "embodiment": str(runtime.embodiment),
        "image_layout": str(runtime.image_layout),
        "action_horizon": int(runtime.chunk_size),
        "action_dim": int(runtime.action_dim),
        "state_dim": int(runtime.state_dim),
        # 真实反映策略 RTC 能力（PyTorch 权重路径无 RTC 实现时为 False，
        # 避免客户端误以为 RTC 生效而静默退化）
        "rtc_capable": bool(getattr(runtime.policy, "_has_rtc_support", False)),
        "config_name": str(args.config_name),
        "architecture": dict(getattr(runtime, "architecture", {}) or {}),
    }

    if srv_logger:
        srv_logger.log_model_loaded(
            model_name="pi05",
            ckpt_path=str(ckpt.resolve()),
            load_time_sec=load_sec,
            extra={
                "embodiment": str(runtime.embodiment),
                "action_horizon": int(runtime.chunk_size),
                "action_dim": int(runtime.action_dim),
                "state_dim": int(runtime.state_dim),
                "architecture": dict(getattr(runtime, "architecture", {}) or {}),
                "rtc_capable": bool(getattr(runtime.policy, "_has_rtc_support", False)),
            },
        )

    kemo_memory = KemoMemory(max_length=int(args.kemo_memory_length)) if args.kemo_memory_length else None
    if kemo_memory is not None and int(runtime.chunk_size) != 10:
        raise SystemExit(
            f"KEMO-style server requires action_horizon=10, got {runtime.chunk_size}"
        )
    if kemo_memory is not None:
        metadata["kemo_memory"] = True
        metadata["kemo_memory_length"] = int(kemo_memory.max_length)
        metadata["kemo_action_dim"] = 16
    server = WebsocketPolicyServer(
        Pi05Policy(runtime, kemo_memory=kemo_memory),
        host=str(args.host),
        port=int(args.port),
        metadata=metadata,
        srv_logger=srv_logger,
    )
    LOGGER.info(
        "OpenPI π₀.₅ 服务端就绪 embodiment=%s，等客户端连 ws://%s:%s",
        runtime.embodiment,
        args.host,
        args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
