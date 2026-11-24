#!/usr/bin/env python3
"""数据集开环 VALIDATION：不接真机。

读 LeRobot parquet + 三路视频，走与真机同一份 YAML 的 websocket，
把预测 chunk 和 GT ``action[t:t+H]``（16D 绝对关节）比 MAE。

用法:
  PYTHONPATH=infer python recipes/common/dataset_validate.py \\
    --config infer/infer/config/lerobot/pi05_00086_4gpu_bs384_ah100_step22445_ws.yaml \\
    --n-samples 24
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
INFER_ROOT = REPO / "infer"
COMMON = Path(__file__).resolve().parent
for _p in (INFER_ROOT, COMMON):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from e2e_ws_client import (  # noqa: E402
    _CAM_TO_OPENPI,
    _action_array,
    _policy,
    pair_check,
)
from infer.policies._lingbot_ws_client import WebsocketClientPolicy  # noqa: E402

DEFAULT_DATASETS = {
    "00086": Path("/home/xk/dataset/00086_20260827_RM_01_YJ_ZL_21"),
    "00036": Path("/home/xk/dataset/00036_20260804_RM_01_yj_v21"),
    "00073": Path("/home/xk/dataset/00073_20260813_RM_01_yj_zl_21"),
}

_CAM_TO_VIDEO = {
    "left": "observation.images.left",
    "front": "observation.images.front",
    "base": "observation.images.front",
    "right": "observation.images.right",
    "cam_left": "observation.images.left",
    "cam_mid": "observation.images.front",
    "cam_right": "observation.images.right",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"YAML 不是 dict: {path}")
    return raw


def _task_id(*texts: str) -> str | None:
    for t in texts:
        m = re.search(r"(000\d{2})", t)
        if m:
            return m.group(1)
    return None


def _info(root: Path) -> dict[str, Any]:
    p = root / "meta" / "info.json"
    if not p.is_file():
        raise SystemExit(f"不是 LeRobot 根（缺 meta/info.json）: {root}")
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(template: str, episode_index: int, chunks_size: int, video_key: str | None = None) -> str:
    chunk = episode_index // max(int(chunks_size), 1)
    kw: dict[str, Any] = {
        "episode_chunk": chunk,
        "episode_index": episode_index,
    }
    if video_key is not None:
        kw["video_key"] = video_key
    return template.format(**kw)


def _read_hwc(path: Path, frame_idx: int, size: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频 {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise RuntimeError(f"{path} 读不到 frame={frame_idx}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if size > 0 and (rgb.shape[0] != size or rgb.shape[1] != size):
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _pi05_obs(pol: dict[str, Any], images_hwc: dict[str, np.ndarray], state: np.ndarray) -> dict[str, Any]:
    names = [str(x) for x in (pol.get("openpi_camera_names") or ["left", "front", "right"])]
    images = {}
    for name in names:
        key = _CAM_TO_OPENPI.get(name.strip().lower(), name)
        hwc = images_hwc[name.strip().lower()]
        images[key] = np.ascontiguousarray(np.transpose(hwc, (2, 0, 1)))
    return {
        "obs": {
            "state": np.asarray(state, dtype=np.float32).reshape(-1),
            "images": images,
            "prompt": str(pol.get("task_prompt") or "null"),
        }
    }


def _walloss_obs(pol: dict[str, Any], images_hwc: dict[str, np.ndarray], state: np.ndarray) -> dict[str, Any]:
    keys = list(pol.get("camera_keys") or [
        "observation.images.front",
        "observation.images.left",
        "observation.images.right",
    ])
    alias = {
        "observation.images.front": "front",
        "observation.images.left": "left",
        "observation.images.right": "right",
        "observation.cam_mid": "front",
        "observation.cam_left": "left",
        "observation.cam_right": "right",
    }
    frame = {}
    for k in keys:
        src = alias.get(k, k.split(".")[-1])
        hwc = images_hwc.get(src)
        if hwc is None:
            raise RuntimeError(f"WALL 缺相机 {k} / {src}")
        frame[k] = hwc
    n_exec = int(pol.get("n_action_steps_override") or 8)
    payload: dict[str, Any] = {
        "obs": [frame],
        "state": np.asarray(state, dtype=np.float32).reshape(-1),
        "n_exec": n_exec,
        "prompt": [str(pol.get("task_prompt") or "")],
    }
    ik = pol.get("ik")
    if isinstance(ik, dict) and ik:
        payload["ik"] = dict(ik)
    return payload


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    n = min(len(pred), len(gt))
    if n == 0:
        return {"n": 0}
    d = min(pred.shape[-1], gt.shape[-1], 16)
    pred, gt = pred[:n, :d], gt[:n, :d]
    err = np.abs(pred - gt)
    sq = (pred - gt) ** 2
    out: dict[str, Any] = {
        "n": n,
        "mae_all": float(err.mean()),
        "mse_all": float(sq.mean()),
        "mae_step0": float(err[0].mean()),
        "mse_step0": float(sq[0].mean()),
        "per_dim_mae": err.mean(axis=0).tolist(),
        "per_dim_mae_step0": err[0].tolist(),
    }
    if d >= 16:
        out.update({
            "mae_left7": float(err[:, 0:7].mean()),
            "mae_right7": float(err[:, 7:14].mean()),
            "mae_grip2": float(err[:, 14:16].mean()),
            "mae_left7_step0": float(err[0, 0:7].mean()),
            "mae_right7_step0": float(err[0, 7:14].mean()),
            "mae_grip2_step0": float(err[0, 14:16].mean()),
        })
    return out


def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    xs = [r[key] for r in rows if key in r and r[key] is not None]
    return float(np.mean(xs)) if xs else None


def main() -> int:
    p = argparse.ArgumentParser(description="数据集开环 VALIDATION（不接真机）")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", default="", help="LeRobot 根；空则按 YAML 里的 000xx 猜本机路径")
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--episode", type=int, default=None, help="只评这一集；默认均匀抽")
    p.add_argument("--t0", type=int, default=None, help="配合 --episode 指定帧")
    p.add_argument("--frac", type=float, default=0.4, help="抽帧位置占 episode 比例")
    p.add_argument("--out", default="", help="写 summary json；空则只打 stdout")
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()

    yaml_path = Path(args.config).expanduser().resolve()
    raw = _load_yaml(yaml_path)
    pol = _policy(raw)
    problems = pair_check(raw, yaml_path)
    if problems:
        print("PAIR_FAIL " + " | ".join(problems), file=sys.stderr)
        return 2

    tid = _task_id(str(yaml_path), str(pol.get("ckpt_dir") or ""), str(raw.get("experiment_name") or ""))
    ds = Path(args.dataset).expanduser() if args.dataset else None
    if ds is None:
        if tid and tid in DEFAULT_DATASETS:
            ds = DEFAULT_DATASETS[tid]
        else:
            raise SystemExit("给 --dataset，或 YAML 里带 00036/00073/00086")
    if not ds.is_dir():
        raise SystemExit(f"数据集不存在: {ds}")
    info = _info(ds)
    chunks = int(info.get("chunks_size") or 1000)
    data_tpl = str(info["data_path"])
    video_tpl = str(info["video_path"])
    n_ep = int(info.get("total_episodes") or 0)
    if n_ep <= 0:
        n_ep = len(list((ds / "data").rglob("episode_*.parquet")))
    ptype = str(pol.get("type") or "").strip().lower()
    size = 224 if ptype == "pi05" else 0
    if ptype == "pi05":
        size = int(pol.get("openpi_image_size") or 224)
        cam_names = [str(x).strip().lower() for x in (pol.get("openpi_camera_names") or ["left", "front", "right"])]
    else:
        cam_names = ["front", "left", "right"]

    if args.episode is not None:
        picks = [(int(args.episode), args.t0)]
    else:
        n = max(1, int(args.n_samples))
        eps = np.unique(np.linspace(0, max(n_ep - 1, 0), num=min(n, n_ep), dtype=int))
        picks = [(int(e), None) for e in eps]

    host = str(pol.get("lingbot_host") or "127.0.0.1")
    port = int(pol.get("lingbot_port"))
    expected = str(pol.get("lingbot_expected_model")).strip()
    client = WebsocketClientPolicy(host=host, port=port, wait_timeout_sec=args.timeout)
    meta = client.get_server_metadata() or {}
    got = str(meta.get("model") or "").strip()
    if got != expected:
        print(f"HANDSHAKE_FAIL expected={expected!r} got={got!r}", file=sys.stderr)
        return 3

    samples: list[dict[str, Any]] = []
    for ep, t_override in picks:
        pq = ds / _fmt(data_tpl, ep, chunks)
        if not pq.is_file():
            print(f"skip missing parquet ep={ep} {pq}", file=sys.stderr)
            continue
        df = pd.read_parquet(pq, columns=["observation.state", "action", "frame_index"])
        n_frames = len(df)
        if n_frames < 2:
            continue
        if t_override is None:
            t = int(np.clip(round((n_frames - 1) * float(args.frac)), 0, n_frames - 2))
        else:
            t = int(np.clip(t_override, 0, n_frames - 2))
        state = np.asarray(df["observation.state"].iloc[t], dtype=np.float32).reshape(-1)
        gt_all = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in df["action"].iloc[t:].tolist()])
        images_hwc: dict[str, np.ndarray] = {}
        for name in cam_names:
            vkey = _CAM_TO_VIDEO.get(name, f"observation.images.{name}")
            vpath = ds / _fmt(video_tpl, ep, chunks, video_key=vkey)
            images_hwc[name] = _read_hwc(vpath, t, size)
        if ptype == "pi05":
            payload = _pi05_obs(pol, images_hwc, state)
            client.infer({"reset": True})
        else:
            payload = _walloss_obs(pol, images_hwc, state)
            client.infer({"reset": True, "prompt": [str(pol.get("task_prompt") or "")]})
        t0 = time.perf_counter()
        reply = client.infer(payload)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        pred = _action_array(reply)
        met = _metrics(pred, gt_all)
        row = {
            "episode": ep,
            "t": t,
            "n_frames": n_frames,
            "pred_shape": list(pred.shape),
            "infer_ms": round(infer_ms, 1),
            **met,
        }
        samples.append(row)
        print(
            f"ep={ep:04d} t={t:04d} pred={tuple(pred.shape)} "
            f"mae_step0={met.get('mae_step0')} L7={met.get('mae_left7_step0')} "
            f"R7={met.get('mae_right7_step0')} grip={met.get('mae_grip2_step0')} "
            f"ms={infer_ms:.0f}",
            flush=True,
        )

    if not samples:
        print("VAL_FAIL 没有成功样本", file=sys.stderr)
        return 4

    summary = {
        "yaml": str(yaml_path),
        "dataset": str(ds),
        "task_id": tid,
        "type": ptype,
        "expected_model": expected,
        "handshake": {k: meta.get(k) for k in ("model", "action_horizon", "action_dim")},
        "n_ok": len(samples),
        "mae_step0": _mean_key(samples, "mae_step0"),
        "mae_all": _mean_key(samples, "mae_all"),
        "mae_left7_step0": _mean_key(samples, "mae_left7_step0"),
        "mae_right7_step0": _mean_key(samples, "mae_right7_step0"),
        "mae_grip2_step0": _mean_key(samples, "mae_grip2_step0"),
        "mae_left7": _mean_key(samples, "mae_left7"),
        "mae_right7": _mean_key(samples, "mae_right7"),
        "mae_grip2": _mean_key(samples, "mae_grip2"),
        "infer_ms_mean": _mean_key(samples, "infer_ms"),
        "note": (
            "开环：pred chunk vs parquet action[t:t+H]（16D 绝对关节 rad / 爪[0,1]）。"
            "step0 是 chunk 第一步，最接近真机会执行的那一拍。"
        ),
        "samples": samples,
    }
    compact = {k: v for k, v in summary.items() if k != "samples"}
    compact["samples_brief"] = [
        {
            "episode": r["episode"],
            "t": r["t"],
            "mae_step0": r.get("mae_step0"),
            "mae_left7_step0": r.get("mae_left7_step0"),
            "mae_right7_step0": r.get("mae_right7_step0"),
            "mae_grip2_step0": r.get("mae_grip2_step0"),
            "infer_ms": r.get("infer_ms"),
        }
        for r in samples
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    print(
        f"VAL_OK n={len(samples)} mae_step0={summary['mae_step0']:.4f} "
        f"L7={summary['mae_left7_step0']:.4f} R7={summary['mae_right7_step0']:.4f} "
        f"grip={summary['mae_grip2_step0']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
