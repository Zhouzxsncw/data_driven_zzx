#!/usr/bin/env python3
"""PI0.5 websocket 握手冒烟验证（合成观测，不动机器人）。

验证: handshake 元数据（model/rtc_capable/ckpt）→ infer 请求返回 (horizon,16)。
默认模式只验握手与基础推理；--rtc 模式从第二包起携带 prev_action_chunk/
executed_steps/rtc 字段，验证服务端 rtc_used=True（PyTorch/JAX 两条路径均适用）。
用法: <venv python> test_pi05_client.py [--host 127.0.0.1] [--port 1242] [--rtc]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "infer"))
sys.path.insert(0, str(_ROOT / "recipes"))

import numpy as np
from infer.policies._lingbot_msgpack_numpy import Packer, unpackb
from infer.policies.openpi_pi05_runtime import (
    _resize_with_pad_numpy,
    _to_hwc_uint8,
    openpi_image_key,
)
import websockets


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1242)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--rtc", action="store_true",
                   help="从第二包起携带 prev_action_chunk/executed_steps/rtc，验证 rtc_used=True")
    args = p.parse_args()

    prompt = (
        "Using both hands, pick up the courier items from the green conveyor belt. "
        "Place the white courier bags into the left box and the brown courier boxes "
        "into the right box."
    )

    async with websockets.connect(f"ws://{args.host}:{args.port}") as ws:
        meta = unpackb(await ws.recv())
        print(f"[handshake] {meta}")
        model = meta.get("model")
        if model != "pi05":
            print(f"FAIL: 期望 model=pi05，得到 {model!r}")
            return 1
        horizon = int(meta.get("action_horizon", 50))
        action_dim = int(meta.get("action_dim", 16))
        state_dim = int(meta.get("state_dim", action_dim))
        embodiment = str(meta.get("embodiment") or "tongbot")
        if args.rtc and not meta.get("rtc_capable"):
            print("FAIL: 服务端 rtc_capable=False，RTC 不可用（检查模型路径是否挂接 RTC 实现）")
            return 1
        if embodiment == "libero":
            cam_names = ("front", "left")
        else:
            cam_names = ("left", "front", "right")

        def make_obs(i: int):
            images = {}
            for idx, cam in enumerate(cam_names):
                img = np.full((480, 640, 3), (i * 40 + idx * 60) % 255, dtype=np.uint8)
                hwc = _resize_with_pad_numpy(_to_hwc_uint8(img), args.image_size, args.image_size)
                # 服务端 openpi_client.image_tools 期望 CHW float32
                images[openpi_image_key(cam)] = np.transpose(hwc, (2, 0, 1)).astype(np.float32)
            state = np.zeros(state_dim, dtype=np.float32)
            return images, state

        ok = True
        rtc_cfg = {
            "enabled": True,
            "prefix_attention_schedule": "exp",
            "max_guidance_weight": 5.0,
            "debug": False,
        }
        prev_actions = None
        for i in range(3):
            images, state = make_obs(i)
            payload = {"obs": {"prompt": prompt, "state": state, "images": images}}
            if args.rtc and prev_actions is not None:
                payload["prev_action_chunk"] = np.asarray(prev_actions, dtype=np.float32)
                payload["executed_steps"] = min(15, max(1, horizon // 3))
                payload["rtc"] = rtc_cfg
            t0 = time.perf_counter()
            await ws.send(Packer().pack(payload))
            resp = unpackb(await ws.recv())
            a = np.asarray(resp["actions"])
            dt = (time.perf_counter() - t0) * 1000
            rtc_used = resp.get("rtc_used")
            print(f"[infer #{i}] actions={a.shape} dtype={a.dtype} "
                  f"range=[{a.min():.4f}, {a.max():.4f}] rtc_used={rtc_used} "
                  f"server_ms={dt:.0f}")
            if a.shape != (horizon, action_dim):
                ok = False
                print(f"  FAIL: 期望 ({horizon}, {action_dim})")
            if args.rtc and i >= 1 and rtc_used is not True:
                ok = False
                print("  FAIL: --rtc 模式下第 2 包起应 rtc_used=True")
            if not np.all(np.isfinite(a)):
                ok = False
                print("  FAIL: actions 含 NaN/Inf")
            prev_actions = a

        print("PASS: PI0.5 websocket 握手+推理验证通过" + ("（含 RTC 生效）" if args.rtc else "")
              if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
