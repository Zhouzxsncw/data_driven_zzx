#!/usr/bin/env python3
"""对比真机 inference obs（关节 + 图像）与训练集 Task1，不含 gripper 反馈。"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[2]
_LEROBOT_SRC = Path(os.environ.get("LEROBOT_SRC") or os.environ.get("CQTONG_SRC") or (_REPO / "train" / "src"))
for _p in (_REPO, _LEROBOT_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.offline.offline_pi05_replay_real_run import CAM_FILES, VideoIndex, build_video_indices  # noqa: E402

JOINT_DIMS = 7  # ljoint1..7，不含 gripper
TASK1_INDEX = 1


def load_train_task1_joints(dataset_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (global_index[N], joints[N,7])，仅 task_index==1。"""
    indices: List[int] = []
    joints: List[np.ndarray] = []
    for pf in sorted((dataset_root / "data").glob("**/*.parquet")):
        t = pq.read_table(pf, columns=["index", "task_index", "observation.state"])
        for idx, ti, st in zip(
            t.column("index").to_pylist(),
            t.column("task_index").to_pylist(),
            t.column("observation.state").to_pylist(),
        ):
            if int(ti) != TASK1_INDEX:
                continue
            j = np.asarray(st, dtype=np.float32)[:JOINT_DIMS]
            indices.append(int(idx))
            joints.append(j)
    return np.asarray(indices, dtype=np.int64), np.stack(joints, axis=0)


def nearest_joint(train_j: np.ndarray, query: np.ndarray) -> Tuple[int, float]:
    diff = train_j - query.reshape(1, -1)
    dist = np.linalg.norm(diff, axis=1)
    i = int(np.argmin(dist))
    return i, float(dist[i])


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_mse(a: np.ndarray, b: np.ndarray, size: Tuple[int, int] = (160, 120)) -> float:
  w, h = size
  a_s = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
  b_s = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
  return float(np.mean((a_s - b_s) ** 2))


def load_train_images(dataset_root: Path, global_index: int) -> Tuple[np.ndarray, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id=f"local/{dataset_root.name}",
        root=str(dataset_root),
        video_backend="pyav",
    )
    item = ds[int(global_index)]
    left = item["observation.cam_left"].detach().cpu().numpy().transpose(1, 2, 0)
    mid = item["observation.cam_mid"].detach().cpu().numpy().transpose(1, 2, 0)
    if left.max() <= 1.0:
        left = (left * 255.0).clip(0, 255)
    if mid.max() <= 1.0:
        mid = (mid * 255.0).clip(0, 255)
    return left.astype(np.uint8), mid.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--image-samples", type=int, default=12, help="做图像 MSE 的 inference 帧数")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    traj = pd.read_parquet(run_dir / "trajectory_data" / "trajectory.parquet")
    step0 = traj[traj["action_step_index"] == 0].sort_values("inference_frame_index")

    real_idx = step0["inference_frame_index"].astype(int).to_numpy()
    real_joints = np.stack(
        [np.asarray(s, dtype=np.float32)[:JOINT_DIMS] for s in step0["observation.state"]]
    )
    real_task = step0["task_segment_index"].fillna(-1).astype(int).to_numpy()

    print("[train] loading task1 joints from parquet ...")
    train_global_idx, train_joints = load_train_task1_joints(dataset_root)
    print(f"[train] task1 frames={len(train_joints)}")

    # 关节分布
    real_mean, real_std = real_joints.mean(0), real_joints.std(0)
    train_mean, train_std = train_joints.mean(0), train_joints.std(0)
    joint_names = [f"ljoint{i+1}" for i in range(JOINT_DIMS)]

    # 每帧最近邻
    rows = []
    nn_dists = []
    for fi, task, qj in zip(real_idx, real_task, real_joints):
        ni, dist = nearest_joint(train_joints, qj)
        nn_dists.append(dist)
        rows.append({
            "inference_frame_index": int(fi),
            "task_segment_index": int(task),
            "real_joints": qj.round(4).tolist(),
            "nn_train_index": int(train_global_idx[ni]),
            "nn_joint_l2": round(dist, 4),
            "nn_joints": train_joints[ni].round(4).tolist(),
            "joint_delta": (qj - train_joints[ni]).round(4).tolist(),
        })

    # 图像对比：均匀抽样
    sample_pos = np.linspace(0, len(rows) - 1, min(args.image_samples, len(rows)), dtype=int)
    eval_dir = run_dir / "eval_images"
    video_idx = build_video_indices(run_dir)
    ds_cache = {}

    def get_train_imgs(gidx: int):
        if gidx not in ds_cache:
            ds_cache[gidx] = load_train_images(dataset_root, gidx)
        return ds_cache[gidx]

    img_rows = []
    try:
        for pos in sample_pos:
            r = rows[int(pos)]
            fi = r["inference_frame_index"]
            row_traj = step0[step0["inference_frame_index"] == fi].iloc[0]
            image_ts = float(row_traj["image_timestamp"])

            real_left = eval_dir / f"frame_{fi:06d}_cam_left.jpg"
            real_mid = eval_dir / f"frame_{fi:06d}_cam_mid.jpg"
            if real_left.is_file():
                rl, rm = load_rgb(real_left), load_rgb(real_mid)
            else:
                rl = video_idx["cam_left"].read_rgb(image_ts)
                rm = video_idx["cam_mid"].read_rgb(image_ts)

            tl, tm = get_train_imgs(r["nn_train_index"])
            img_rows.append({
                "inference_frame_index": fi,
                "nn_train_index": r["nn_train_index"],
                "nn_joint_l2": r["nn_joint_l2"],
                "mse_cam_left": round(resize_mse(rl, tl), 2),
                "mse_cam_mid": round(resize_mse(rm, tm), 2),
            })
            print(
                f"  img fi={fi} nn={r['nn_train_index']} joint_l2={r['nn_joint_l2']:.3f} "
                f"mse_left={img_rows[-1]['mse_cam_left']:.1f} mse_mid={img_rows[-1]['mse_cam_mid']:.1f}"
            )
    finally:
        for v in video_idx.values():
            v.close()

    summary = {
        "real_frames": len(rows),
        "train_task1_frames": len(train_joints),
        "joint_l2": {
            "mean": round(float(np.mean(nn_dists)), 4),
            "median": round(float(np.median(nn_dists)), 4),
            "min": round(float(np.min(nn_dists)), 4),
            "max": round(float(np.max(nn_dists)), 4),
            "p90": round(float(np.percentile(nn_dists, 90)), 4),
        },
        "real_joint_mean": real_mean.round(4).tolist(),
        "real_joint_std": real_std.round(4).tolist(),
        "train_joint_mean": train_mean.round(4).tolist(),
        "train_joint_std": train_std.round(4).tolist(),
        "per_joint_mean_abs_delta": np.abs(real_mean - train_mean).round(4).tolist(),
        "image_mse": img_rows,
        "frames": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # markdown
    lines = [
        "# 真机 vs 训练 Task1 观测对比（不含 gripper）",
        "",
        "## 关节最近邻距离（7 维 ljoint，真机 125 inference 帧）",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 训练 Task1 帧数 | {len(train_joints)} |",
        f"| L2 mean | {summary['joint_l2']['mean']} |",
        f"| L2 median | {summary['joint_l2']['median']} |",
        f"| L2 min / max | {summary['joint_l2']['min']} / {summary['joint_l2']['max']} |",
        f"| L2 p90 | {summary['joint_l2']['p90']} |",
        "",
        "## 关节边际分布（mean ± std）",
        "",
        "| 关节 | 真机 mean±std | 训练 Task1 mean±std | |mean差| |",
        "|------|---------------|-------------------|--------|",
    ]
    for i, name in enumerate(joint_names):
        lines.append(
            f"| {name} | {real_mean[i]:.3f}±{real_std[i]:.3f} | "
            f"{train_mean[i]:.3f}±{train_std[i]:.3f} | {abs(real_mean[i]-train_mean[i]):.3f} |"
        )
    lines += [
        "",
        "## 图像 MSE（真机 eval_images vs 关节最近邻训练帧，resize 160×120）",
        "",
        "| infer_frame | nn_train_idx | joint_L2 | MSE left | MSE mid |",
        "|-------------|--------------|----------|----------|---------|",
    ]
    for ir in img_rows:
        lines.append(
            f"| {ir['inference_frame_index']} | {ir['nn_train_index']} | "
            f"{ir['nn_joint_l2']:.3f} | {ir['mse_cam_left']:.1f} | {ir['mse_cam_mid']:.1f} |"
        )
    if img_rows:
        ml = np.mean([r["mse_cam_left"] for r in img_rows])
        mm = np.mean([r["mse_cam_mid"] for r in img_rows])
        lines += [
            "",
            f"图像 MSE 均值：cam_left **{ml:.1f}**，cam_mid **{mm:.1f}**（0=完全相同，>500 差异较大）",
        ]
    lines += ["", f"完整 JSON：`{args.output_json}`"]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {args.output_json}")
    print(f"[saved] {args.output_md}")


if __name__ == "__main__":
    main()
