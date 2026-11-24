#!/usr/bin/env python3
"""Open-loop WALL-OSS on local 00086 EE dataset (no hardware).

Loads the jiangsuan-16 SFT ckpt, runs flow matching on recorded frames,
and compares predicted relative EE (and the unapplied absolute EE) to GT.
Optionally: relative EE → absolute EE → continuous IK → 16D joints.

  source recipes/walloss/env_infer.sh
  python recipes/walloss/openloop_00086.py --n-frames 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if not (SCRIPTS / "ee_infer_pipeline.py").is_file():
    SCRIPTS = Path("/home/xk/VLA/wall_oss_ppu_setup/scripts")
sys.path.insert(0, str(SCRIPTS))

from convert_tongbot_joint_to_ee import _stack_object_column  # noqa: E402
from ee_infer_pipeline import (  # noqa: E402
    ee20_to_absolute,
    ee20_to_relative,
    geodesic_deg,
    rot6d_to_matrix,
)
from walloss_policy import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_PROMPT,
    WallOssTongPolicy,
    to_hwc_uint8,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("walloss_openloop")

DEFAULT_EE_ROOT = Path("/data/wall-oss/datasets/00086_rm75_ee")
CAM_KEYS = (
    "observation.images.front",
    "observation.images.left",
    "observation.images.right",
)


def _episode_parquet(root: Path, ep: int) -> Path:
    files = sorted((root / "data").rglob(f"episode_{ep:06d}.parquet"))
    if not files:
        raise FileNotFoundError(f"episode {ep} not in {root}")
    return files[0]


def _video_path(root: Path, cam: str, ep: int) -> Path:
    files = sorted((root / "videos").rglob(f"{cam}/episode_{ep:06d}.mp4"))
    if not files:
        raise FileNotFoundError(f"video {cam} episode {ep} not in {root}")
    return files[0]


def _read_frame(path: Path, t: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise RuntimeError(f"frame {t} missing in {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return to_hwc_uint8(rgb)


def _rot_err_deg(a: np.ndarray, b: np.ndarray) -> float:
    return geodesic_deg(rot6d_to_matrix(a), rot6d_to_matrix(b))


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    left_pos = np.linalg.norm(pred[:, 0:3] - gt[:, 0:3], axis=1)
    right_pos = np.linalg.norm(pred[:, 10:13] - gt[:, 10:13], axis=1)
    left_rot = np.array([_rot_err_deg(pred[i, 3:9], gt[i, 3:9]) for i in range(n)])
    right_rot = np.array([_rot_err_deg(pred[i, 13:19], gt[i, 13:19]) for i in range(n)])
    grip = np.abs(pred[:, [9, 19]] - gt[:, [9, 19]])
    return {
        "n": n,
        "left_pos_mae_m": float(left_pos.mean()),
        "right_pos_mae_m": float(right_pos.mean()),
        "left_pos_max_m": float(left_pos.max()),
        "right_pos_max_m": float(right_pos.max()),
        "left_rot_mae_deg": float(left_rot.mean()),
        "right_rot_mae_deg": float(right_rot.mean()),
        "grip_mae": float(grip.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--ee-root", type=Path, default=DEFAULT_EE_ROOT)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--t0", type=int, default=10)
    p.add_argument("--n-frames", type=int, default=4)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--n-exec", type=int, default=8)
    p.add_argument("--instruction", default=DEFAULT_PROMPT)
    p.add_argument("--skip-ik", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    pq = _episode_parquet(args.ee_root, args.episode)
    df = pd.read_parquet(pq)
    state = _stack_object_column(df["observation.state"])
    state_ee = _stack_object_column(df["observation.state.ee"])
    action_ee = _stack_object_column(df["action.ee"])
    vids = {cam: _video_path(args.ee_root, cam, args.episode) for cam in CAM_KEYS}

    policy = WallOssTongPolicy(
        args.ckpt,
        instruction=args.instruction,
        n_exec=args.n_exec,
        action_horizon=args.horizon,
    )

    reports = []
    t = args.t0
    for k in range(args.n_frames):
        if t + args.horizon >= len(state_ee):
            break
        images = {cam: _read_frame(path, t) for cam, path in vids.items()}
        gt_rel = ee20_to_relative(action_ee[t : t + args.horizon], state_ee[t])
        pred_rel = policy.predict_rel_ee(images, state_ee[t], args.instruction)
        pred_abs = ee20_to_absolute(pred_rel, state_ee[t])
        gt_abs = action_ee[t : t + args.horizon]
        row = {
            "k": k,
            "t": t,
            "rel": _metrics(pred_rel, gt_rel),
            "abs": _metrics(pred_abs, gt_abs),
        }
        if not args.skip_ik:
            joints, info = policy.predict_logical16(
                images,
                state[t],
                args.instruction,
                n_exec=args.n_exec,
                rel_ee=pred_rel,
            )
            gt_j = state[t + 1 : t + 1 + len(joints)]
            if len(gt_j) == len(joints):
                err = np.linalg.norm(
                    np.rad2deg(joints[:, :14] - gt_j[:, :14]), axis=1
                )
                row["ik"] = {
                    "n_planned": info["n_planned"],
                    "aborted_ik": info["aborted_ik"],
                    "joint14_mae_deg": float(err.mean()) if err.size else None,
                    "max_step_deg": info.get("max_step_deg"),
                }
            else:
                row["ik"] = {
                    "n_planned": info["n_planned"],
                    "aborted_ik": info["aborted_ik"],
                    "max_step_deg": info.get("max_step_deg"),
                }
        reports.append(row)
        LOGGER.info(
            "frame t=%s rel=%s abs=%s ik=%s",
            t,
            row["rel"],
            row["abs"],
            row.get("ik"),
        )
        t += args.stride

    summary = {
        "ckpt": str(args.ckpt),
        "episode": str(pq),
        "n": len(reports),
        "left_pos_mae_m": float(np.mean([r["rel"]["left_pos_mae_m"] for r in reports]))
        if reports
        else None,
        "right_pos_mae_m": float(np.mean([r["rel"]["right_pos_mae_m"] for r in reports]))
        if reports
        else None,
        "abs_left_pos_mae_m": float(np.mean([r["abs"]["left_pos_mae_m"] for r in reports]))
        if reports
        else None,
        "abs_right_pos_mae_m": float(
            np.mean([r["abs"]["right_pos_mae_m"] for r in reports])
        )
        if reports
        else None,
        "left_rot_mae_deg": float(np.mean([r["rel"]["left_rot_mae_deg"] for r in reports]))
        if reports
        else None,
        "right_rot_mae_deg": float(
            np.mean([r["rel"]["right_rot_mae_deg"] for r in reports])
        )
        if reports
        else None,
        "frames": reports,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
