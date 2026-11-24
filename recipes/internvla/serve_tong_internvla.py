#!/usr/bin/env python3
"""InternVLA-A1.5 TongBot websocket server, VA-client compatible.

Speaks the same msgpack protocol as LingBot-VA:
  handshake metadata  {model: internvla_a15, ...}
  reset:              {reset: True, prompt: [str]}
  compute_kv_cache:   ignored
  infer:              {obs: [cam_dict, ...], state: (16,)}
  reply:              {action: (T, 16)}  # T = chunk_size (50)

00036 微调 ckpt：action/state 统计是 16D TongBot 关节；模型内部 pad 到 32D。
相机顺序与 tong_dual_arm schema 一致：front / left / right。
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

CODE = os.environ.get("INTERNVLA_CODE", "/home/xk/VLA/InternVLA-A-series")
sys.path.insert(0, os.path.join(CODE, "src"))
sys.path.insert(0, CODE)

import msgpack
import numpy as np
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("internvla_serve")

DEFAULT_CKPT = (
    "/data/internvla/outputs/00036_finetune_bs32/checkpoints/002000/pretrained_model"
)
DEFAULT_VLM = "/data/internvla/ckpts/Qwen3.5-2B"
DEFAULT_PROMPT = (
    "With both hands, pick up the yellow banana and the yellow lemon and put them "
    "into the black box; then take the green vegetables and place them into the blue box."
)
CAM_FRONT = "observation.images.front"
CAM_LEFT = "observation.images.left"
CAM_RIGHT = "observation.images.right"
TONGBOT_DIM = 16


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


def _pad_state16(state) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.size < TONGBOT_DIM:
        pad = np.zeros(TONGBOT_DIM, dtype=np.float32)
        pad[: arr.size] = arr
        return pad
    return arr[:TONGBOT_DIM]


def _to_tongbot_16d(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim == 1:
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


class InternVlaTongPolicy:
    def __init__(self, backend, instruction: str):
        self.backend = backend
        self.instruction = instruction

    def reset(self, prompt: str | None) -> None:
        if prompt:
            self.instruction = str(prompt).strip() or self.instruction
        if hasattr(self.backend, "policy") and hasattr(self.backend.policy, "reset"):
            self.backend.policy.reset()
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
        example = {
            "image": [
                _to_hwc_uint8(last[CAM_FRONT]),
                _to_hwc_uint8(last[CAM_LEFT]),
                _to_hwc_uint8(last[CAM_RIGHT]),
            ],
            "state": _pad_state16(msg.get("state")),
            "lang": self.instruction,
        }
        t0 = time.perf_counter()
        out = self.backend.infer({"examples": [example]})
        actions = _to_tongbot_16d(out["actions"])
        LOGGER.info(
            "infer action=%s sec=%.2f prompt=%s",
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


def build_backend(ckpt: str, vlm: str, device: str, robot_type: str):
    from evaluation.LIBERO.policy_server.backends.policy_backend_internvla_a1_5 import (
        InternVLAA15Backend,
    )

    LOGGER.info("loading InternVLA-A1.5 ckpt=%s vlm=%s robot=%s", ckpt, vlm, robot_type)
    return InternVLAA15Backend(
        ckpt_path=ckpt,
        device=device,
        robot_type=robot_type,
        vlm_model_path=vlm,
        action_loss_only=True,
        inference_backend="standard",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.environ.get("INTERNVLA_CKPT", DEFAULT_CKPT))
    p.add_argument("--vlm", default=os.environ.get("VLM_MODEL_PATH", DEFAULT_VLM))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=1237)
    p.add_argument("--device", default="cuda")
    p.add_argument("--robot-type", default="tong_dual_arm")
    p.add_argument("--instruction", default=DEFAULT_PROMPT)
    p.add_argument("--load-only", action="store_true")
    args = p.parse_args()

    if not os.path.isdir(args.ckpt):
        raise SystemExit(f"ckpt 不存在: {args.ckpt}")

    backend = build_backend(args.ckpt, args.vlm, args.device, args.robot_type)
    chunk = int(getattr(backend, "chunk_size", 50))
    LOGGER.info("model ready chunk_size=%s action_dim=%s", chunk, backend.action_space()["dim"])
    if args.load_only:
        LOGGER.info("load-only ok")
        return

    policy = InternVlaTongPolicy(backend, args.instruction)
    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={
            "model": "internvla_a15",
            "ckpt": args.ckpt,
            "action_dim": TONGBOT_DIM,
            "action_horizon": chunk,
            "robot_type": args.robot_type,
            "rtc_capable": False,
        },
    ).serve_forever()


if __name__ == "__main__":
    main()
