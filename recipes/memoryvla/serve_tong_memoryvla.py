#!/usr/bin/env python3
"""MemoryVLA TongBot websocket server, VA-client compatible.

Speaks the same msgpack protocol as LingBot-VA:
  handshake metadata  {model: memoryvla, ...}
  reset:              {reset: True, prompt: [str]}
  compute_kv_cache:   ignored (memory bank resets on episode_first_frame)
  infer:              {obs: [cam_dict, ...], state: (16,)}
  reply:              {action: (T, 16)}  # T = future_action_window_size+1 = 16

00086 训练：front → image，left → wrist_image，action_dim=16。
官方 deploy.py 只吃单张图；这里用 front 作为 primary（与 convert_00086 一致）。
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http
import logging
import math
import os
import sys
import time
import traceback

import yaml

ROOT = os.environ.get("MEMVLA_ROOT", "/home/xk/VLA/MemoryVLA")
sys.path.insert(0, ROOT)

import msgpack
import numpy as np
import websockets
import websockets.asyncio.server as _server
import websockets.frames
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("memoryvla_serve")

DEFAULT_CKPT = (
    "/data/memoryvla/outputs/00086/memvla_00086_full--image_aug/checkpoints/"
    "step-008000-epoch-00-loss=0.0228.pt"
)
DEFAULT_PROMPT = (
    "Using both hands, pull open the drawer with the blue handle using your left hand, "
    "take out the white medicine box with your right hand, place it into the green drawer, "
    "and finally close the drawer."
)
CAM_FRONT = "observation.images.front"
TONGBOT_DIM = 16
HORIZON = 16


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


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        raise ValueError(f"期望 HWC/CHW 图像，得到 shape={arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        max_v = float(np.nanmax(arr)) if arr.size else 1.0
        if max_v <= 1.0 + 1e-3:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def _scale_and_resize(image: Image.Image, target_size=(224, 224), scale=0.9):
    w, h = image.size
    new_w = int(w * math.sqrt(scale))
    new_h = int(h * math.sqrt(scale))
    margin_w = (w - new_w) // 2
    margin_h = (h - new_h) // 2
    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    return image.resize(target_size, resample=Image.LANCZOS)


def resize_image(image: Image.Image, size=(224, 224)) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize(size, resample=Image.LANCZOS)
    return _scale_and_resize(image, target_size=size, scale=0.9)


def _to_tongbot_16d(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"期望 action (T, D)，得到 {arr.shape}")
    if arr.shape[-1] < TONGBOT_DIM:
        pad = np.zeros((arr.shape[0], TONGBOT_DIM), dtype=np.float32)
        pad[:, : arr.shape[-1]] = arr
        arr = pad
    else:
        arr = arr[:, :TONGBOT_DIM]
    return np.ascontiguousarray(arr)


def _load_run_cfg(ckpt: str) -> dict:
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        LOGGER.warning("没有 config.yaml: %s", cfg_path)
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class MemoryVlaTongPolicy:
    def __init__(self, vla, instruction: str, unnorm_key: str, cfg_scale: float, num_ddim_steps: int):
        self.vla = vla
        self.instruction = instruction
        self.unnorm_key = unnorm_key
        self.cfg_scale = cfg_scale
        self.num_ddim_steps = num_ddim_steps
        self._first = True

    def reset(self, prompt: str | None) -> None:
        if prompt:
            self.instruction = str(prompt).strip() or self.instruction
        self._first = True
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
        if CAM_FRONT not in last:
            raise KeyError(f"obs 缺相机 {CAM_FRONT}；keys={list(last)}")
        image = Image.fromarray(_to_hwc_uint8(last[CAM_FRONT]))
        image = resize_image(image, size=(224, 224))
        first = "True" if self._first else "False"
        t0 = time.perf_counter()
        unnormed, _ = self.vla.predict_action(
            image=image,
            instruction=self.instruction,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale,
            use_ddim=True,
            num_ddim_steps=self.num_ddim_steps,
            episode_first_frame=first,
        )
        self._first = False
        actions = _to_tongbot_16d(unnormed)
        LOGGER.info(
            "infer first=%s action=%s sec=%.2f prompt=%s",
            first,
            tuple(actions.shape),
            time.perf_counter() - t0,
            self.instruction[:80],
        )
        return {"action": actions}


class WebsocketPolicyServer:
    def __init__(self, policy, host: str, port: int, metadata: dict | None = None):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
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
        LOGGER.info("client %s connected", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                obs = unpackb(await websocket.recv())
                t0 = time.monotonic()
                action = self._policy.infer(obs)
                if not isinstance(action, dict):
                    action = {"action": action}
                action["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                LOGGER.info("client %s closed", websocket.remote_address)
                break
            except Exception:
                tb = traceback.format_exc()
                LOGGER.exception("infer failed")
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


def build_model(ckpt: str, use_bf16: bool):
    from vla import load_vla

    run_cfg = _load_run_cfg(ckpt)
    kwargs = {
        "action_dim": int(run_cfg.get("action_dim", TONGBOT_DIM)),
        "future_action_window_size": int(run_cfg.get("future_action_window_size", HORIZON - 1)),
        "action_model_type": str(run_cfg.get("action_model_type", "DiT-L")),
        "use_ema": bool(run_cfg.get("use_ema", False)),
        "dataloader_type": str(run_cfg.get("dataloader_type", "stream")),
        "group_size": int(run_cfg.get("group_size", 16)),
        "per_token_size": int(run_cfg.get("per_token_size", 256)),
        "mem_length": int(run_cfg.get("mem_length", 256)),
        "retrieval_layers": int(run_cfg.get("retrieval_layers", 2)),
        "use_timestep_pe": bool(run_cfg.get("use_timestep_pe", True)),
        "fusion_type": str(run_cfg.get("fusion_type", "gate")),
        "consolidate_type": str(run_cfg.get("consolidate_type", "tome")),
        "update_fused": bool(run_cfg.get("update_fused", False)),
        "use_bf16": use_bf16,
    }
    LOGGER.info("loading MemoryVLA ckpt=%s kwargs=%s", ckpt, {k: kwargs[k] for k in ("action_dim", "future_action_window_size", "mem_length")})
    vla = load_vla(model_id_or_path=ckpt, load_for_training=False, **kwargs)
    vla = vla.to("cuda").eval()
    if use_bf16:
        vla = vla.to(__import__("torch").bfloat16)
    else:
        vla = vla.to(__import__("torch").float32)
    return vla, kwargs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.environ.get("MEMVLA_CKPT", DEFAULT_CKPT))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=1238)
    p.add_argument("--instruction", default=DEFAULT_PROMPT)
    p.add_argument("--unnorm-key", default="custom_finetuning")
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--num-ddim-steps", type=int, default=10)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--load-only", action="store_true")
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"ckpt 不存在: {args.ckpt}")

    vla, kwargs = build_model(args.ckpt, args.bf16)
    horizon = int(kwargs["future_action_window_size"]) + 1
    LOGGER.info("model ready action_dim=%s horizon=%s", kwargs["action_dim"], horizon)
    if args.load_only:
        LOGGER.info("load-only ok")
        return

    policy = MemoryVlaTongPolicy(
        vla,
        args.instruction,
        unnorm_key=args.unnorm_key,
        cfg_scale=args.cfg_scale,
        num_ddim_steps=args.num_ddim_steps,
    )
    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={
            "model": "memoryvla",
            "ckpt": args.ckpt,
            "action_dim": TONGBOT_DIM,
            "action_horizon": horizon,
            "unnorm_key": args.unnorm_key,
            "rtc_capable": False,
        },
    ).serve_forever()


if __name__ == "__main__":
    main()
