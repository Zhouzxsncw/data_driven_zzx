#!/usr/bin/env python3
"""PI0-FAST websocket 推理端到端验证（合成观测，不动机器人）。

验证: handshake 元数据 → infer 请求 → 返回 (15,16) actions 数值合理。
用法: <jax venv python> test_pi0fast_client.py [--host 127.0.0.1] [--port 1243]
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
    p.add_argument("--port", type=int, default=1243)
    p.add_argument("--image-size", type=int, default=224, help="模型输入分辨率（协议契约）")
    args = p.parse_args()

    prompt = (
        "With both hands, pick up the yellow banana and the yellow lemon and put them "
        "into the black box; then take the green vegetables and place them into the blue box."
    )
    # 合成观测: 3 相机 480x640 HWC uint8 + 16D state（取自 00036 数据集合理关节范围）
    rng = np.random.default_rng(0)
    raw_images = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
    state = np.array(
        [0.12, 0.55, -0.30, 0.45, 0.10, 0.20, 0.35,
         -0.10, 0.50, 0.28, -0.40, -0.15, 0.25, -0.30, 0.5, 0.6],
        dtype=np.float32,
    )

    # 按真实客户端协议构造观测（pi05_adapter → runtime.build_observation）:
    # resize_with_pad 224x224 → CHW uint8 → openpi 相机键名。
    # 服务端 _decode_aloha 会按 CHW→HWC 还原，务必不能直接发 HWC 原图。
    image_dict = {}
    for cam, img in zip(("left", "front", "right"), raw_images):
        hwc = _resize_with_pad_numpy(_to_hwc_uint8(img), args.image_size, args.image_size)
        image_dict[openpi_image_key(cam)] = np.transpose(hwc, (2, 0, 1))

    packer = Packer()
    uri = f"ws://{args.host}:{args.port}"
    async with websockets.connect(uri, max_size=None, compression=None) as ws:
        meta = unpackb(await ws.recv())
        print("[handshake]", {k: meta.get(k) for k in ("model", "action_horizon", "action_dim", "rtc_capable", "ckpt")})
        assert meta.get("model") == "pi0_fast", f"期望 pi0_fast, got {meta.get('model')}"
        assert int(meta.get("action_dim")) == 16

        obs = {
            "state": state,
            "images": image_dict,
            "prompt": prompt,
        }
        for i in range(3):
            t0 = time.perf_counter()
            await ws.send(packer.pack({"obs": obs}))
            reply = unpackb(await ws.recv())
            dt = time.perf_counter() - t0
            actions = np.asarray(reply["actions"])
            print(
                f"[infer #{i}] actions={actions.shape} dtype={actions.dtype} "
                f"range=[{actions.min():.4f}, {actions.max():.4f}] "
                f"server_ms={reply.get('server_timing', {}).get('infer_ms', -1):.0f} wall={dt*1000:.0f}ms"
            )
            assert actions.shape == (15, 16), f"期望 (15,16), got {actions.shape}"
            assert np.isfinite(actions).all(), "actions 含 NaN/Inf"

        # reset 协议
        await ws.send(packer.pack({"reset": True}))
        reply = unpackb(await ws.recv())
        print("[reset]", reply)
        assert reply.get("reset") is True

    print("PASS: PI0-FAST websocket 端到端验证通过")


if __name__ == "__main__":
    asyncio.run(main())
