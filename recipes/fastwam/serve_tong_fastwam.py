#!/usr/bin/env python3
"""FastWAM TongBot websocket server, VA-client compatible.

Protocol (same as LingBot-VA / Cosmos3 / WALL-OSS):
  handshake metadata  {model: fastwam, ...}
  reset:              {reset: True, prompt: [str]}
  compute_kv_cache:   ignored
  infer:              {obs: [cam_dict, ...], state: (16,)}
  reply:              {action: (action_horizon, 16)}

Multi-camera preprocessing mirrors training concat_multi_camera="robotwin":
  front → 256×320  |  left → 128×160  |  right → 128×160
  concat → [3, 384, 320]  →  normalize [-1, 1]
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

# 共享服务端请求级日志模块（recipes/_server_logger.py）
_THIS_DIR = Path(__file__).resolve().parent
_RECIPES_DIR = _THIS_DIR.parent
if str(_RECIPES_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPES_DIR))
from _server_logger import ServerRequestLogger  # noqa: E402

import msgpack
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("fastwam_serve")

CAM_FRONT = "observation.images.front"
CAM_LEFT = "observation.images.left"
CAM_RIGHT = "observation.images.right"
TONGBOT_DIM = 16


# ---------------------------------------------------------------------------
# msgpack helpers (same as other serve scripts)
# ---------------------------------------------------------------------------

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


def _state16(state) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float64).reshape(-1)
    if arr.size < TONGBOT_DIM:
        pad = np.zeros(TONGBOT_DIM, dtype=np.float64)
        pad[: arr.size] = arr
        return pad
    return arr[:TONGBOT_DIM]


# ---------------------------------------------------------------------------
# Robotwin-style multi-camera preprocessing
# ---------------------------------------------------------------------------

def _hwc_to_chw_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC numpy → float32 CHW tensor [0, 255]."""
    if img.ndim == 4:
        img = img[0]
    arr = np.ascontiguousarray(img, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]


def preprocess_robotwin(obs_last: dict) -> torch.Tensor:
    """Three-camera → [1, 3, 384, 320] float32 tensor in [-1, 1]."""
    cam_front = _hwc_to_chw_tensor(obs_last[CAM_FRONT])
    cam_left = _hwc_to_chw_tensor(obs_last[CAM_LEFT])
    cam_right = _hwc_to_chw_tensor(obs_last[CAM_RIGHT])

    # Robotwin layout: top=front(256×320), bottom=left|right(128×160 each)
    cam_top = transforms_F.resize(
        cam_front, size=[256, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
    )
    cam_l = transforms_F.resize(
        cam_left, size=[128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
    )
    cam_r = transforms_F.resize(
        cam_right, size=[128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
    )
    bottom = torch.cat([cam_l, cam_r], dim=-1)  # [C, 128, 320]
    combined = torch.cat([cam_top, bottom], dim=-2)  # [C, 384, 320]

    # Normalize to [-1, 1]
    combined = combined / 255.0
    combined = combined * 2.0 - 1.0
    return combined.unsqueeze(0)  # [1, C, 384, 320]


# ---------------------------------------------------------------------------
# FastWAM model loader
# ---------------------------------------------------------------------------

def _build_fastwam_model(args):
    """Create FastWAM model and load fine-tuned checkpoint."""
    from fastwam.models.wan22.fastwam import FastWAM

    video_dit_config = {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1.0e-06,
        "seperated_timestep": True,
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": False,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": 16,
        "action_group_causal_mask_mode": "group_diagonal",
    }
    action_dit_config = {
        "action_dim": 16,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1.0e-06,
        "use_gradient_checkpointing": False,
    }

    LOGGER.info("Loading Wan2.2-TI2V-5B components...")
    model = FastWAM.from_wan22_pretrained(
        device=args.device,
        torch_dtype=torch.bfloat16,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len=128,
        load_text_encoder=True,
        proprio_dim=16,
        redirect_common_files=False,
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=args.action_dit_backbone,
        skip_dit_load_from_pretrain=False,
        mot_checkpoint_mixed_attn=False,
        video_train_shift=5.0,
        video_infer_shift=5.0,
        video_num_train_timesteps=1000,
        action_train_shift=1.0,
        action_infer_shift=1.0,
        action_num_train_timesteps=1000,
    )

    if args.ckpt:
        LOGGER.info("loading FastWAM ckpt=%s device=%s", args.ckpt, args.device)
        model.load_checkpoint(args.ckpt)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

def _load_action_denorm_stats(stats_path: str):
    """Load z-score denormalization params from dataset_stats.json."""
    import json
    with open(stats_path, "r") as f:
        stats = json.load(f)
    action_stats = stats["action"]["default"]
    mean = np.array(action_stats["global_mean"], dtype=np.float32)
    std = np.array(action_stats["global_std"], dtype=np.float32)
    LOGGER.info("loaded action denorm stats: mean=%s std=%s", mean[:4], std[:4])
    return mean, std


class FastWAMServePolicy:
    def __init__(self, model, instruction: str, action_horizon: int, denoise: int,
                 action_mean: np.ndarray | None = None,
                 action_std: np.ndarray | None = None,
                 compile_action_infer: bool = False):
        self.model = model
        self.instruction = instruction
        self.action_horizon = action_horizon
        self.denoise = denoise
        self._compile = compile_action_infer
        self._action_mean = action_mean
        self._action_std = action_std
        self._context = None
        self._context_mask = None
        self._warmup_done = False
        self._encode_prompt(instruction)

    def _encode_prompt(self, prompt: str) -> None:
        # Wrap with training-time template to match dataset distribution
        wrapped = PROMPT_TEMPLATE.format(task=prompt) if not prompt.startswith("A video recorded") else prompt
        LOGGER.info("encoding prompt: %s", wrapped[:160])
        ctx, mask = self.model.encode_prompt(wrapped)
        self._context = ctx.detach()
        self._context_mask = mask.detach()

    def reset(self, prompt: str | None) -> None:
        if prompt:
            self.instruction = str(prompt).strip() or self.instruction
            self._encode_prompt(self.instruction)
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
        for key in (CAM_FRONT, CAM_LEFT, CAM_RIGHT):
            if key not in last:
                raise KeyError(f"obs 缺相机 {key}；keys={list(last)}")

        # --- timing: preprocess ---
        t_pre = time.perf_counter()
        input_image = preprocess_robotwin(last).to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        joints = _state16(msg.get("state"))
        proprio = torch.from_numpy(joints).unsqueeze(0).to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        preprocess_ms = (time.perf_counter() - t_pre) * 1000

        # --- timing: model forward ---
        t_model = time.perf_counter()
        out = self.model.infer_action(
            prompt=None,
            input_image=input_image,
            action_horizon=self.action_horizon,
            proprio=proprio,
            context=self._context,
            context_mask=self._context_mask,
            num_inference_steps=self.denoise,
            compile_action_infer=self._compile,
        )
        model_ms = (time.perf_counter() - t_model) * 1000
        if self._compile and not self._warmup_done:
            LOGGER.info("torch.compile warmup call done (%.1fms)", model_ms)
            self._warmup_done = True

        # --- timing: postprocess ---
        t_post = time.perf_counter()
        actions = out["action"].numpy()  # [action_horizon, 16] float32 (z-score normalized)
        # Z-score denormalization: raw = z * std + mean
        if self._action_mean is not None and self._action_std is not None:
            actions = actions * self._action_std + self._action_mean
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        postprocess_ms = (time.perf_counter() - t_post) * 1000

        total_ms = preprocess_ms + model_ms + postprocess_ms
        LOGGER.info(
            "infer action=%s denoise=%d total=%.1fms (pre=%.1f model=%.1f post=%.1f) range=[%.3f,%.3f]",
            actions.shape, self.denoise, total_ms,
            preprocess_ms, model_ms, postprocess_ms,
            actions.min(), actions.max(),
        )
        return {
            "action": actions,
            "_timing": {
                "preprocess_ms": preprocess_ms,
                "model_ms": model_ms,
                "postprocess_ms": postprocess_ms,
            },
        }


# ---------------------------------------------------------------------------
# Websocket server (identical to other serve scripts)
# ---------------------------------------------------------------------------

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
            self._srv_logger.log_connection(client_addr=client_addr, event="connect",
                                            metadata=self._metadata)
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

                # 提取内部 timing 分解（policy 可能返回 _timing）
                timing = result.pop("_timing", None)
                action = result
                action["server_timing"] = {"infer_ms": infer_ms}

                # 网络发送
                t_send = time.monotonic()
                await websocket.send(packer.pack(action))
                send_ms = (time.monotonic() - t_send) * 1000

                if self._srv_logger:
                    # 合并 timing：preprocess + model + postprocess + network_send
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

DEFAULT_CKPT = "/home/xk/VLA/FastWAM/runs/tongbot_00073_3cam384_1e-4/2026-09-07_19-14-02/checkpoints/weights/step_020000.pt"
DEFAULT_BACKBONE = "/home/xk/VLA/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
DEFAULT_DATASET_STATS = "/home/xk/VLA/FastWAM/runs/tongbot_00073_3cam384_1e-4/2026-09-07_19-14-02/dataset_stats.json"
DEFAULT_PROMPT = (
    "Using both hands, pick up the courier items from the green conveyor belt. "
    "Place the white courier bags into the left box and the brown courier boxes into the right box."
)
# Must match training-time template in robot_video_dataset.py DEFAULT_PROMPT
PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.environ.get("FASTWAM_CKPT", DEFAULT_CKPT))
    p.add_argument("--action-dit-backbone", default=os.environ.get("FASTWAM_BACKBONE", DEFAULT_BACKBONE))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("FASTWAM_PORT", "1240")))
    p.add_argument("--instruction", default=os.environ.get("FASTWAM_PROMPT", DEFAULT_PROMPT))
    p.add_argument("--action-horizon", type=int, default=int(os.environ.get("FASTWAM_ACTION_HORIZON", "32")))
    p.add_argument("--denoise", type=int, default=int(os.environ.get("FASTWAM_DENOISE", "10")))
    p.add_argument("--device", default=os.environ.get("FASTWAM_DEVICE", "cuda"))
    p.add_argument("--dataset-stats", default=os.environ.get("FASTWAM_DATASET_STATS", DEFAULT_DATASET_STATS))
    p.add_argument("--compile", action="store_true",
                   default=bool(os.environ.get("FASTWAM_COMPILE", "")),
                   help="启用 torch.compile(mode=reduce-overhead) 编译 action denoise 路径，首次推理有编译开销")
    p.add_argument("--load-only", action="store_true")
    p.add_argument("--log-dir", default=os.environ.get("FASTWAM_LOG_DIR", ""),
                   help="请求级 JSONL 日志目录（默认与 serve 脚本同目录）")
    p.add_argument("--no-request-log", action="store_true",
                   help="禁用请求级 JSONL 日志")
    args = p.parse_args()

    load_t0 = time.time()
    model = _build_fastwam_model(args)
    load_sec = time.time() - load_t0
    LOGGER.info("model ready action_horizon=%d denoise=%d compile=%s (%.1fs)",
                args.action_horizon, args.denoise, args.compile, load_sec)

    # Load z-score denormalization stats
    action_mean, action_std = None, None
    if args.dataset_stats and Path(args.dataset_stats).is_file():
        action_mean, action_std = _load_action_denorm_stats(args.dataset_stats)
    else:
        LOGGER.warning("dataset_stats not found at %s; actions will NOT be denormalized!", args.dataset_stats)

    # 请求级日志
    srv_logger = None
    if not args.no_request_log:
        log_dir = Path(args.log_dir) if args.log_dir else _THIS_DIR
        log_path = log_dir / "fastwam_requests.jsonl"
        srv_logger = ServerRequestLogger(log_path, gpu_device=0, model_name="fastwam")
        LOGGER.info("请求级日志: %s", log_path)

    if args.load_only:
        LOGGER.info("load-only OK")
        if srv_logger:
            srv_logger.close()
        return

    if srv_logger:
        srv_logger.log_model_loaded(
            model_name="fastwam",
            ckpt_path=str(Path(args.ckpt).resolve()),
            load_time_sec=load_sec,
            extra={
                "action_horizon": args.action_horizon,
                "denoise": args.denoise,
                "action_dim": 16,
            },
        )

    policy = FastWAMServePolicy(model, args.instruction, args.action_horizon, args.denoise,
                                action_mean=action_mean, action_std=action_std,
                                compile_action_infer=args.compile)
    meta = {
        "model": "fastwam",
        "ckpt": str(Path(args.ckpt).resolve()),
        "action_horizon": args.action_horizon,
        "denoise": args.denoise,
        "compile": args.compile,
        "action_dim": 16,
        "rtc_capable": False,
    }
    LOGGER.info("server listening on %s:%s (compile=%s)", args.host, args.port, args.compile)
    server = WebsocketPolicyServer(policy, args.host, args.port, metadata=meta,
                                   srv_logger=srv_logger)
    try:
        server.serve_forever()
    finally:
        if srv_logger:
            summary = srv_logger.close()
            LOGGER.info("服务端统计: %d 请求 (%d 错误), 运行 %.1fs, 有效 FPS: %s, 日志: %s",
                        summary["total_requests"], summary.get("error_count", 0),
                        summary["duration_sec"],
                        summary.get("infer_ms_stats", {}).get("mean", "N/A"),
                        summary["log_path"])


if __name__ == "__main__":
    main()
