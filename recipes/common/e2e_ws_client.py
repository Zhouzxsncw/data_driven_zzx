#!/usr/bin/env python3
"""本机 websocket 冒烟客户端：和真机共用同一份 YAML，不接相机/臂。

核对：
  1) YAML 里的 lingbot_expected_model / port / ckpt 与服务端握手一致
  2) π0.5 TongBot 必须 [7,7,-1,-1] 且 adapt_to_pi=false
  3) 合成（或数据集）观测走一遍 infer，动作维数对上

用法:
  PYTHONPATH=infer python recipes/common/e2e_ws_client.py \\
    --config infer/infer/config/lerobot/pi05_00086_4gpu_bs384_ah100_step22445_ws.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
INFER_ROOT = REPO / "infer"
if str(INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(INFER_ROOT))

from infer.policies._lingbot_ws_client import WebsocketClientPolicy  # noqa: E402

_CAM_TO_OPENPI = {
    "left": "left_wrist_0_rgb",
    "front": "base_0_rgb",
    "base": "base_0_rgb",
    "right": "right_wrist_0_rgb",
    "cam_left": "left_wrist_0_rgb",
    "cam_mid": "base_0_rgb",
    "cam_right": "right_wrist_0_rgb",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"YAML 不是 dict: {path}")
    return raw


def _policy(raw: dict[str, Any]) -> dict[str, Any]:
    pol = raw.get("policy")
    if not isinstance(pol, dict):
        raise SystemExit("YAML 缺少 policy:")
    return pol


def _gray(h: int = 224, w: int = 224) -> np.ndarray:
    return np.full((h, w, 3), 114, dtype=np.uint8)


def pair_check(raw: dict[str, Any], yaml_path: Path) -> list[str]:
    """静态对齐检查。返回问题列表（空=过）。"""
    pol = _policy(raw)
    problems: list[str] = []
    ptype = str(pol.get("type") or "").strip().lower()
    expected = str(pol.get("lingbot_expected_model") or "").strip()
    ckpt = Path(str(pol.get("ckpt_dir") or "")).expanduser()
    if not expected:
        problems.append("policy.lingbot_expected_model 为空")
    if ptype == "pi05":
        if str(pol.get("pi05_backend") or "") != "websocket":
            problems.append("π0.5 本机 e2e 需要 policy.pi05_backend: websocket（与 launch_pi05_server 同一份 YAML）")
        dims = pol.get("openpi_delta_action_dims")
        if list(dims or []) != [7, 7, -1, -1]:
            problems.append(f"openpi_delta_action_dims={dims}，TongBot 必须 [7, 7, -1, -1]")
        if pol.get("openpi_adapt_to_pi") not in (False, "false", 0, None):
            if pol.get("openpi_adapt_to_pi") is True:
                problems.append("openpi_adapt_to_pi=true，TongBot 必须 false")
        weight = ckpt / "model.safetensors"
        params = ckpt / "params"
        if not weight.is_file() and not params.exists():
            problems.append(f"权重不存在: {ckpt}")
        stats = ckpt.parent / "assets" / "norm_stats.json"
        if ckpt.name == "pretrained_model":
            stats = ckpt.parent / "assets" / "norm_stats.json"
        else:
            stats = ckpt / "assets" / "norm_stats.json"
        if not stats.is_file():
            problems.append(f"缺少 norm_stats.json（ckpt 同级 assets/）: {stats}")
    elif ptype == "walloss":
        if not (ckpt / "model.safetensors").is_file():
            problems.append(f"WALL-OSS 权重不存在: {ckpt}/model.safetensors")
        env_ckpt = str(__import__("os").environ.get("WALLOSS_CKPT") or "").strip()
        if env_ckpt and Path(env_ckpt).expanduser().resolve() != ckpt.resolve():
            problems.append(
                f"WALLOSS_CKPT={env_ckpt} 与 YAML ckpt_dir={ckpt} 不一致（服务端/客户端会各吃一份）"
            )
    else:
        problems.append(f"e2e 目前只接 pi05 / walloss，得到 type={ptype}（{yaml_path}）")
    return problems


def _pi05_payload(pol: dict[str, Any]) -> dict[str, Any]:
    """和 Pi05Adapter 真机路径一样：CHW uint8 + 包在 obs 里。"""
    names = [str(x) for x in (pol.get("openpi_camera_names") or ["left", "front", "right"])]
    size = int(pol.get("openpi_image_size") or 224)
    images = {}
    for name in names:
        key = _CAM_TO_OPENPI.get(name.strip().lower(), name)
        hwc = np.ascontiguousarray(_gray(size, size))
        images[key] = np.ascontiguousarray(np.transpose(hwc, (2, 0, 1)))  # CHW
    state_dim = int(pol.get("openpi_state_dim") or pol.get("openpi_action_dim") or 16)
    return {
        "obs": {
            "state": np.zeros((state_dim,), dtype=np.float32),
            "images": images,
            "prompt": str(pol.get("task_prompt") or "null"),
        }
    }


def _walloss_payload(pol: dict[str, Any]) -> dict[str, Any]:
    keys = list(pol.get("camera_keys") or [
        "observation.images.front",
        "observation.images.left",
        "observation.images.right",
    ])
    frame = {k: _gray(480, 640) for k in keys}
    n_exec = int(pol.get("n_action_steps_override") or 8)
    payload: dict[str, Any] = {
        "obs": [frame],
        "state": np.zeros((16,), dtype=np.float32),
        "n_exec": n_exec,
        "prompt": [str(pol.get("task_prompt") or "")],
    }
    ik = pol.get("ik")
    if isinstance(ik, dict) and ik:
        payload["ik"] = dict(ik)
    return payload


def _action_array(reply: Any) -> np.ndarray:
    if not isinstance(reply, dict):
        raise RuntimeError(f"服务端回了非 dict: {type(reply)}")
    for key in ("actions", "action"):
        if key in reply:
            return np.asarray(reply[key])
    raise RuntimeError(f"回包没有 actions/action，keys={sorted(reply)}")


def main() -> int:
    p = argparse.ArgumentParser(description="本机 YAML 对齐 + websocket 冒烟（不接真机）")
    p.add_argument("--config", required=True, help="与 launch_*_server / launch_client 同一份 YAML")
    p.add_argument("--n", type=int, default=2, help="infer 轮数")
    p.add_argument("--pair-only", action="store_true", help="只做静态对齐，不连服务")
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    yaml_path = Path(args.config).expanduser().resolve()
    raw = _load_yaml(yaml_path)
    pol = _policy(raw)
    problems = pair_check(raw, yaml_path)
    print(json.dumps({
        "yaml": str(yaml_path),
        "type": pol.get("type"),
        "expected_model": pol.get("lingbot_expected_model"),
        "port": pol.get("lingbot_port"),
        "ckpt_dir": pol.get("ckpt_dir"),
        "pair_ok": not problems,
        "problems": problems,
    }, ensure_ascii=False, indent=2))
    if problems:
        print("PAIR_FAIL", file=sys.stderr)
        return 2
    print("PAIR_OK")
    if args.pair_only:
        return 0

    host = str(pol.get("lingbot_host") or "127.0.0.1")
    port = int(pol.get("lingbot_port"))
    expected = str(pol.get("lingbot_expected_model")).strip()
    client = WebsocketClientPolicy(host=host, port=port, wait_timeout_sec=args.timeout)
    meta = client.get_server_metadata() or {}
    got = str(meta.get("model") or "").strip()
    print("handshake", json.dumps(meta, default=str, ensure_ascii=False))
    if got != expected:
        print(f"HANDSHAKE_FAIL expected={expected!r} got={got!r}", file=sys.stderr)
        return 3
    print(f"HANDSHAKE_OK model={got} horizon={meta.get('action_horizon')} dim={meta.get('action_dim')}")

    ptype = str(pol.get("type") or "").strip().lower()
    if ptype == "pi05":
        payload = _pi05_payload(pol)
        client.infer({"reset": True})
    else:
        payload = _walloss_payload(pol)
        client.infer({"reset": True, "prompt": [str(pol.get("task_prompt") or "")]})

    last_shape = None
    for i in range(max(1, args.n)):
        reply = client.infer(payload)
        act = _action_array(reply)
        last_shape = tuple(act.shape)
        timing = reply.get("server_timing") if isinstance(reply, dict) else None
        print(f"infer#{i} action={last_shape} dtype={act.dtype} timing={timing}")
        if act.ndim != 2 or act.shape[-1] != 16:
            print(f"SHAPE_FAIL 期望 (T,16) 得到 {last_shape}", file=sys.stderr)
            return 4
    print(f"E2E_OK action_shape={last_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
