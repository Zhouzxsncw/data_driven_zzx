#!/usr/bin/env python3
"""GigaWorld-Policy-0.5 TongBot websocket server, VA-client compatible.

Protocol (same as LingBot-VA / FastWAM / Motus):
  handshake metadata  {model: gwp05, ...}
  reset:              {reset: True, prompt: [str]}
  compute_kv_cache:   ignored
  infer:              {obs: [cam_dict, ...], state: (16,)}
  reply:              {action: (48, 16)}  已反归一化、delta 维已加回当前关节

TongBot 与官方 AgileX open-loop 的差异（必须对齐 H100 微调）：
  - 相机 front/left/right，T 形拼图 320×384
  - action_dim=16（发布权重 in_action_channels=16，不能 pad 32）
  - 14 关节 delta + 2 夹爪绝对
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http
import logging
import os
import sys
import time
import traceback
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_RECIPES_DIR = _THIS_DIR.parent
if str(_RECIPES_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPES_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _server_logger import ServerRequestLogger  # noqa: E402
from gwp_tongbot import (  # noqa: E402
    ACTION_CHUNK,
    CAM_KEYS,
    DEFAULT_DENOISE,
    DELTA_MASK,
    DST_SIZE_WH,
    HANDSHAKE_MODEL,
    NUM_VIDEO_FRAMES,
    TONGBOT_DIM,
    delta_to_abs,
    load_norm_stats,
    quantile_denormalize,
    quantile_normalize,
    resolve_transformer_dir,
    state16,
    transformer_weights_ready,
    views_from_obs,
)

from PIL import Image
import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("gwp_serve")

DEFAULT_CKPT = "/home/xk/VLA/h100_ckpts/gwp05_tongbot_00073"
DEFAULT_NORM = "/home/xk/VLA/h100_ckpts/gwp05_tongbot_00073/norm_stats_00073.json"
DEFAULT_PROMPT = (
    "Using both hands, pick up the courier items from the green conveyor belt. "
    "Place the white courier bags into the left box and the brown courier boxes into the right box."
)
GWP_ROOT = Path(os.environ.get("GWP_ROOT", "/home/xk/VLA/giga-world-policy"))


def _setup_gwp_path() -> None:
    root = str(GWP_ROOT)
    third = os.path.join(root, "third_party")
    for path in (
        root,
        os.path.join(third, "giga-datasets"),
        os.path.join(third, "giga-models"),
        os.path.join(third, "giga-train"),
    ):
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def _load_pipeline(args):
    _setup_gwp_path()
    from diffusers.models import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    import importlib.util

    from world_action_model.models import CasualWorldActionTransformer_MoT

    ol_path = GWP_ROOT / "scripts" / "inference_openloop.py"
    spec = importlib.util.spec_from_file_location("gwp_inference_openloop", ol_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"找不到 {ol_path}")
    ol = importlib.util.module_from_spec(spec)
    sys.modules["gwp_inference_openloop"] = ol
    spec.loader.exec_module(ol)
    WAPipeline = ol.WAPipeline
    compile_policy_action_blocks = ol.compile_policy_action_blocks
    get_ref_image_3views = ol.get_ref_image_3views
    resolve_torch_dtype = ol.resolve_torch_dtype

    device = args.device
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    torch_dtype = resolve_torch_dtype(args.model_dtype)
    trans_dir = resolve_transformer_dir(args.ckpt)
    LOGGER.info("transformer=%s base_model=%s dtype=%s", trans_dir, args.base_model, torch_dtype)

    vae = AutoencoderKLWan.from_pretrained(args.base_model, subfolder="vae", torch_dtype=torch_dtype)
    transformer = CasualWorldActionTransformer_MoT.from_pretrained(str(trans_dir), torch_dtype=torch_dtype)
    transformer.eval()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    pipe = WAPipeline.from_pretrained(
        args.base_model,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
    )
    pipe.to(device=device, dtype=torch_dtype)
    pipe._model_dtype_name = str(torch_dtype)
    if args.compile:
        LOGGER.info("torch.compile action-blocks mode=%s", args.compile_mode)
        compiled = compile_policy_action_blocks(
            pipe, mode=args.compile_mode, fullgraph=False, scope="action-blocks"
        )
        LOGGER.info("compiled modules: %s", compiled)
    pipe._gwp_get_ref = get_ref_image_3views
    return pipe


class GWPServePolicy:
    def __init__(self, pipe, instruction: str, norm_stats: dict, device: str, denoise: int):
        self.pipe = pipe
        self.instruction = instruction
        self.device = device
        self.denoise = int(denoise)
        self._state_stats = norm_stats["observation.state"]
        self._action_stats = norm_stats["action"]
        self._chunk_idx = 0

    def reset(self, prompt: str | None) -> None:
        if prompt:
            self.instruction = str(prompt).strip() or self.instruction
        self._chunk_idx = 0
        LOGGER.info("reset prompt=%s", self.instruction[:160])

    def infer(self, msg: dict) -> dict:
        if msg.get("reset"):
            prompt = msg.get("prompt")
            if isinstance(prompt, (list, tuple)) and prompt:
                prompt = prompt[0]
            self.reset(prompt if isinstance(prompt, str) else None)
            return {}
        if msg.get("compute_kv_cache"):
            return {}

        obs_list = msg.get("obs")
        if not obs_list:
            raise KeyError("infer 消息缺少 obs")
        last = obs_list[-1]

        t_pre = time.perf_counter()
        front, left, right = views_from_obs(last)
        pils = [Image.fromarray(front), Image.fromarray(left), Image.fromarray(right)]
        ref_image = self.pipe._gwp_get_ref(pils, DST_SIZE_WH)
        raw_state = state16(msg.get("state"))
        norm_state = quantile_normalize(raw_state, self._state_stats, TONGBOT_DIM)
        state_t = torch.from_numpy(norm_state).to(device=self.device)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        preprocess_ms = (time.perf_counter() - t_pre) * 1000

        t_model = time.perf_counter()
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        pred = self.pipe(
            height=DST_SIZE_WH[1],
            width=DST_SIZE_WH[0],
            action_chunk=ACTION_CHUNK,
            state=state_t,
            num_frames=NUM_VIDEO_FRAMES,
            guidance_scale=0.0,
            num_inference_steps=self.denoise,
            image=ref_image,
            return_dict=False,
            prompt=self.instruction,
            action_dim=TONGBOT_DIM,
        )
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        model_ms = (time.perf_counter() - t_model) * 1000

        t_post = time.perf_counter()
        pred_np = pred.detach().float().cpu().numpy()
        if pred_np.ndim == 3:
            pred_np = pred_np[0]
        pred_phys = quantile_denormalize(pred_np, self._action_stats, TONGBOT_DIM)
        actions = delta_to_abs(pred_phys, raw_state, DELTA_MASK)
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        postprocess_ms = (time.perf_counter() - t_post) * 1000
        self._chunk_idx += 1

        LOGGER.info(
            "infer action=%s denoise=%d total=%.1fms (pre=%.1f model=%.1f post=%.1f) range=[%.3f,%.3f]",
            actions.shape,
            self.denoise,
            preprocess_ms + model_ms + postprocess_ms,
            preprocess_ms,
            model_ms,
            postprocess_ms,
            float(actions.min()),
            float(actions.max()),
        )
        return {
            "action": actions,
            "_timing": {
                "preprocess_ms": preprocess_ms,
                "model_ms": model_ms,
                "postprocess_ms": postprocess_ms,
            },
        }


class WebsocketPolicyServer:
    def __init__(self, policy, host: str, port: int, metadata: dict | None = None,
                 srv_logger: ServerRequestLogger | None = None):
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
            try:
                obs = unpackb(await websocket.recv())
                self._request_id += 1
                t0 = time.monotonic()
                result = self._policy.infer(obs)
                if not isinstance(result, dict):
                    result = {"action": result}
                infer_ms = (time.monotonic() - t0) * 1000
                timing = result.pop("_timing", None)
                action = result
                action["server_timing"] = {"infer_ms": infer_ms}
                t_send = time.monotonic()
                await websocket.send(packer.pack(action))
                send_ms = (time.monotonic() - t_send) * 1000
                if self._srv_logger:
                    log_timing = dict(timing) if timing else {}
                    log_timing["total_ms"] = round(infer_ms, 2)
                    log_timing["network_send_ms"] = round(send_ms, 2)
                    self._srv_logger.log_request(
                        request_id=self._request_id,
                        client_addr=client_addr,
                        infer_ms=infer_ms,
                        request_data=obs if isinstance(obs, dict) else None,
                        response_data=action,
                        timing=log_timing,
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.environ.get("GWP_CKPT", DEFAULT_CKPT))
    p.add_argument(
        "--base-model",
        default=os.environ.get(
            "GWP_WAN_PRETRAINED",
            "/data/giga-world-policy/pretrained/Wan2.2-TI2V-5B-Diffusers",
        ),
    )
    p.add_argument("--norm-stats", default=os.environ.get("GWP_NORM_STATS", DEFAULT_NORM))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("GWP_PORT", "1243")))
    p.add_argument("--instruction", default=os.environ.get("GWP_PROMPT", DEFAULT_PROMPT))
    p.add_argument("--denoise", type=int, default=int(os.environ.get("GWP_DENOISE", str(DEFAULT_DENOISE))))
    p.add_argument("--device", default=os.environ.get("GWP_DEVICE", "cuda:0"))
    p.add_argument("--model-dtype", default="bf16")
    p.add_argument(
        "--compile",
        action="store_true",
        default=bool(os.environ.get("GWP_COMPILE", "")),
        help="torch.compile action-blocks；Blackwell 上默认关",
    )
    p.add_argument("--compile-mode", default="reduce-overhead")
    p.add_argument("--pretrained-only", action="store_true")
    p.add_argument("--load-only", action="store_true")
    p.add_argument("--log-dir", default=os.environ.get("GWP_LOG_DIR", ""))
    p.add_argument("--no-request-log", action="store_true")
    args = p.parse_args()

    if args.pretrained_only:
        args.ckpt = os.environ.get(
            "GWP05_TRANSFORMER_PRETRAINED",
            "/data/giga-world-policy/pretrained/Giga-World-Policy-0.5",
        )
        LOGGER.warning("pretrained-only: 加载发布基座，不是 00073 SFT")

    if not transformer_weights_ready(args.ckpt):
        LOGGER.error(
            "权重不存在: %s\n"
            "推理只要 transformer/（约 12G），不要拉 79G 的 pytorch_model optimizer。\n"
            "  bash recipes/gigaworld_policy/pull_gwp_ckpt.sh\n"
            "远端: /data1/gwp/outputs/00073_gwp05_ft/models/checkpoint_epoch_22_step_50000/transformer",
            args.ckpt,
        )
        sys.exit(1)
    if not Path(args.norm_stats).is_file():
        LOGGER.error("norm stats 不存在: %s", args.norm_stats)
        sys.exit(1)
    if not Path(args.base_model).is_dir():
        LOGGER.error("Wan Diffusers 基座不存在: %s", args.base_model)
        sys.exit(1)

    load_t0 = time.time()
    pipe = _load_pipeline(args)
    load_sec = time.time() - load_t0
    LOGGER.info("model ready action_horizon=%d denoise=%d (%.1fs)", ACTION_CHUNK, args.denoise, load_sec)

    stats = load_norm_stats(args.norm_stats)
    srv_logger = None
    if not args.no_request_log:
        log_dir = Path(args.log_dir) if args.log_dir else _THIS_DIR
        log_path = log_dir / "gwp_requests.jsonl"
        srv_logger = ServerRequestLogger(log_path, gpu_device=0, model_name=HANDSHAKE_MODEL)
        LOGGER.info("请求级日志: %s", log_path)

    if args.load_only:
        LOGGER.info("load-only OK")
        if srv_logger:
            srv_logger.close()
        return

    if srv_logger:
        srv_logger.log_model_loaded(
            model_name=HANDSHAKE_MODEL,
            ckpt_path=str(Path(args.ckpt).resolve()),
            load_time_sec=load_sec,
            extra={
                "action_horizon": ACTION_CHUNK,
                "denoise": args.denoise,
                "action_dim": TONGBOT_DIM,
            },
        )

    policy = GWPServePolicy(pipe, args.instruction, stats, args.device, args.denoise)
    meta = {
        "model": HANDSHAKE_MODEL,
        "ckpt": str(Path(args.ckpt).resolve()),
        "action_horizon": ACTION_CHUNK,
        "denoise": args.denoise,
        "action_dim": TONGBOT_DIM,
        "cameras": list(CAM_KEYS),
        "dst_size_wh": list(DST_SIZE_WH),
        "rtc_capable": False,
    }
    LOGGER.info("server listening on %s:%s", args.host, args.port)
    server = WebsocketPolicyServer(policy, args.host, args.port, metadata=meta, srv_logger=srv_logger)
    try:
        server.serve_forever()
    finally:
        if srv_logger:
            summary = srv_logger.close()
            LOGGER.info(
                "服务端统计: %d 请求 (%d 错误), 运行 %.1fs, 日志: %s",
                summary["total_requests"],
                summary.get("error_count", 0),
                summary["duration_sec"],
                summary["log_path"],
            )


if __name__ == "__main__":
    main()
