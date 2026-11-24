#!/usr/bin/env python3
"""π₀.₅ TongBot 16D 合成观测 probe：直接打服务端 ws://host:port。

- 不开 RealSense / UDP 臂 / 夹爪，只发合成观测确认 action.shape 与 handshake 声明的 (horizon, action_dim) 一致。
- 默认 port=1241（yika step1890）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_INFER_ROOT = Path(os.environ.get("INFER_ROOT", _REPO_ROOT / "infer")).expanduser()
if str(_INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_INFER_ROOT))

from infer.policies.openpi_pi05_runtime import OpenpiPi05Runtime  # noqa: E402
from infer.policies.pi05_embodiment import TONGBOT  # noqa: E402


def _connect_ws(host: str, port: int, timeout: float):
    from infer.policies._lingbot_ws_client import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=host, port=port, wait_timeout_sec=timeout)
    meta = dict(client.get_server_metadata() or {})
    return client, meta


def _build_obs(runtime: OpenpiPi05Runtime, prompt: str) -> dict:
    images = [
        np.zeros((224, 224, 3), dtype=np.uint8),  # left
        np.zeros((224, 224, 3), dtype=np.uint8),  # front (mid)
        np.zeros((224, 224, 3), dtype=np.uint8),  # right
    ]
    state = np.zeros(runtime.state_dim, dtype=np.float32)
    return runtime.build_observation(images, state, prompt)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1241)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--n", type=int, default=1, help="连续发几帧")
    p.add_argument(
        "--prompt",
        default="With both hands, pick up the yellow banana and the yellow lemon and put them into the black box; then take the green vegetables and place them into the blue box.",
    )
    args = p.parse_args()

    runtime = OpenpiPi05Runtime(
        ckpt_dir=Path("/tmp/pi05_tongbot_probe_unused"),
        config_name=TONGBOT.config_name,
        repo_path=None,
        camera_names=list(TONGBOT.cameras),
        action_dim=TONGBOT.action_dim,
        state_dim=TONGBOT.state_dim,
        action_horizon=TONGBOT.action_horizon,
        embodiment="tongbot",
    )
    print(f"[runtime] config={TONGBOT.config_name} action_dim={runtime.action_dim} state_dim={runtime.state_dim} horizon={runtime._action_horizon}")

    client, meta = _connect_ws(args.host, args.port, args.timeout)
    print(f"[handshake] {json.dumps(meta, ensure_ascii=False)}")
    if str(meta.get("model") or "").lower() != "pi05":
        m=meta.get("model"); print(f"FAIL: model={m!r}")
        return 1
    emb = str(meta.get("embodiment") or "")
    if emb and emb != "tongbot":
        print(f"FAIL: 服务端 embodiment={emb!r}，tongbot 客户端不兼容")
        return 1

    obs = _build_obs(runtime, args.prompt)
    for i in range(int(args.n)):
        t0 = time.perf_counter()
        result = client.infer({"obs": obs})
        dt = (time.perf_counter() - t0) * 1000
        actions = result.get("actions")
        if actions is None:
            actions = result.get("action")
        if actions is None:
            print(f"FAIL: 服务端返回无 actions 字段: {list(result.keys())}")
            return 1
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        print(
            f"[infer #{i+1}] actions.shape={tuple(arr.shape)} "
            f"min={arr.min():+.3f} max={arr.max():+.3f} mean={arr.mean():+.3f} "
            f"finite={bool(np.isfinite(arr).all())} roundtrip_ms={dt:.0f}"
        )
        exp_h = int(meta.get("action_horizon") or runtime._action_horizon)
        exp_d = int(meta.get("action_dim") or runtime.action_dim)
        if arr.shape[0] != exp_h or arr.shape[1] != exp_d:
            print(f"FAIL: 期望 ({exp_h}, {exp_d})（以服务端 handshake 为准），得到 {tuple(arr.shape)}")
            return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
