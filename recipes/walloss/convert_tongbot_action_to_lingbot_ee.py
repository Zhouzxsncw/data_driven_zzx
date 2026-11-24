#!/usr/bin/env python3
"""00073：OBS 16D 关节不变，把 action 换成 LingBot-VLA 末端 16D。

复用已有 RealMan FK 结果 ``action.ee``（WALL-OSS 20D：xyz+rot6d+爪），
转成 LingBot ``end.position`` 布局：

    [左 xyz3+quat_xyzw4 | 右 xyz3+quat_xyzw4 | 左爪 | 右爪]

``observation.state`` / 相机 / videos 原样保留。原关节 action 另存 ``action.joint``。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ACTION_EE_NAMES = [
    "left_ee_x",
    "left_ee_y",
    "left_ee_z",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_qw",
    "right_ee_x",
    "right_ee_y",
    "right_ee_z",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_qw",
    "gripper",
    "right_gripper",
]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """WALL-OSS rot6d = ZYX 旋转矩阵前两行 → (..., 3, 3)。"""
    r = np.asarray(rot6d, dtype=np.float64)
    single = r.ndim == 1
    if single:
        r = r[np.newaxis, :]
    r0 = r[:, 0:3]
    r1 = r[:, 3:6]
    r0 = r0 / np.clip(np.linalg.norm(r0, axis=1, keepdims=True), 1e-8, None)
    r1 = r1 - np.sum(r0 * r1, axis=1, keepdims=True) * r0
    r1 = r1 / np.clip(np.linalg.norm(r1, axis=1, keepdims=True), 1e-8, None)
    r2 = np.cross(r0, r1)
    R = np.stack([r0, r1, r2], axis=1)
    return R[0] if single else R


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Shepperd，xyzw，w>=0。与 lingbot ee_pose_transform.matrix_to_quat 一致。"""
    M = np.asarray(R, dtype=np.float64)
    single = M.ndim == 2
    if single:
        M = M[np.newaxis, ...]
    m00, m01, m02 = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    m10, m11, m12 = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    m20, m21, m22 = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]
    trace = m00 + m11 + m22
    q = np.zeros((M.shape[0], 4), dtype=np.float64)

    s = np.sqrt(np.clip(trace + 1.0, 1e-10, None)) * 2.0
    q[:, 3] = 0.25 * s
    q[:, 0] = (m21 - m12) / s
    q[:, 1] = (m02 - m20) / s
    q[:, 2] = (m10 - m01) / s

    cond2 = (m00 > m11) & (m00 > m22) & (trace <= 0)
    s2 = np.sqrt(np.clip(1.0 + m00 - m11 - m22, 1e-10, None)) * 2.0
    q[cond2, 0] = 0.25 * s2[cond2]
    q[cond2, 1] = (m01 + m10)[cond2] / s2[cond2]
    q[cond2, 2] = (m02 + m20)[cond2] / s2[cond2]
    q[cond2, 3] = (m21 - m12)[cond2] / s2[cond2]

    cond3 = (~cond2) & (m11 > m22) & (trace <= 0)
    s3 = np.sqrt(np.clip(1.0 - m00 + m11 - m22, 1e-10, None)) * 2.0
    q[cond3, 0] = (m01 + m10)[cond3] / s3[cond3]
    q[cond3, 1] = 0.25 * s3[cond3]
    q[cond3, 2] = (m12 + m21)[cond3] / s3[cond3]
    q[cond3, 3] = (m02 - m20)[cond3] / s3[cond3]

    cond4 = (~cond2) & (~cond3) & (trace <= 0)
    s4 = np.sqrt(np.clip(1.0 - m00 - m11 + m22, 1e-10, None)) * 2.0
    q[cond4, 0] = (m02 + m20)[cond4] / s4[cond4]
    q[cond4, 1] = (m12 + m21)[cond4] / s4[cond4]
    q[cond4, 2] = 0.25 * s4[cond4]
    q[cond4, 3] = (m10 - m01)[cond4] / s4[cond4]

    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    flip = q[:, 3] < 0
    q[flip] *= -1.0
    return q[0] if single else q


def ee20_to_lingbot16(ee20: np.ndarray) -> np.ndarray:
    """(N, 20) WALL-OSS abs EE → (N, 16) LingBot end+gripper。"""
    arr = np.asarray(ee20, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[1] < 20:
        raise ValueError(f"期望 action.ee 20D，得到 {arr.shape}")
    n = arr.shape[0]
    out = np.empty((n, 16), dtype=np.float32)
    q_l = matrix_to_quat_xyzw(rot6d_to_matrix(arr[:, 3:9]))
    q_r = matrix_to_quat_xyzw(rot6d_to_matrix(arr[:, 13:19]))
    out[:, 0:3] = arr[:, 0:3]
    out[:, 3:7] = q_l
    out[:, 7:10] = arr[:, 10:13]
    out[:, 10:14] = q_r
    out[:, 14] = arr[:, 9]
    out[:, 15] = arr[:, 19]
    return out


def _stack(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in series.to_numpy()], axis=0)


def _rows(arr: np.ndarray) -> list[np.ndarray]:
    return [row.astype(np.float32) for row in arr]


def _feat16() -> dict:
    return {"dtype": "float32", "shape": [16], "names": list(ACTION_EE_NAMES), "fps": 15}


def _feat_joint() -> dict:
    return {
        "dtype": "float32",
        "shape": [16],
        "names": [
            "ljoint1", "ljoint2", "ljoint3", "ljoint4", "ljoint5", "ljoint6", "ljoint7",
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
            "gripper", "right_gripper",
        ],
        "fps": 15,
    }


def _stats(chunks: list[np.ndarray]) -> dict:
    data = np.concatenate(chunks, axis=0)
    return {
        "mean": data.mean(0).tolist(),
        "std": np.clip(data.std(0), 1e-6, None).tolist(),
        "min": data.min(0).tolist(),
        "max": data.max(0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
        "q02": np.quantile(data, 0.02, axis=0).tolist(),
        "q98": np.quantile(data, 0.98, axis=0).tolist(),
        "count": [int(data.shape[0])] * data.shape[1],
    }


def convert(src: Path, dst: Path, *, overwrite: bool, limit: int | None) -> dict:
    info = json.loads((src / "meta" / "info.json").read_text())
    files = sorted((src / "data").rglob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有 parquet: {src / 'data'}")
    if limit is not None:
        files = files[: int(limit)]
    if dst.exists() and not overwrite:
        existing = list((dst / "data").rglob("episode_*.parquet")) if (dst / "data").exists() else []
        if existing:
            raise FileExistsError(f"{dst} 已有 {len(existing)} 个 parquet，加 --overwrite")

    dst.mkdir(parents=True, exist_ok=True)
    acc_action: list[np.ndarray] = []
    acc_state: list[np.ndarray] = []
    n_frames = 0

    for pq in tqdm(files, desc="action.ee 20D → LingBot 16D"):
        rel = pq.relative_to(src)
        out_pq = dst / rel
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(pq)
        if "action.ee" not in df.columns:
            raise KeyError(f"{pq} 没有 action.ee，先跑 convert_tongbot_joint_to_ee.py")
        state = _stack(df["observation.state"])
        joint = _stack(df["action"])
        ee20 = _stack(df["action.ee"])
        act16 = ee20_to_lingbot16(ee20)
        if not np.allclose(act16[:, 14], joint[:, 14], atol=1e-5):
            raise RuntimeError(f"{pq.name} 左爪和 action.ee 对不上")
        df["action.joint"] = _rows(joint)
        df["action"] = _rows(act16)
        df.to_parquet(out_pq, index=False)
        acc_action.append(act16)
        acc_state.append(state)
        n_frames += int(act16.shape[0])

    (dst / "meta").mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl", "modality.json"):
        src_meta = src / "meta" / name
        if src_meta.is_file():
            shutil.copy2(src_meta, dst / "meta" / name)

    new_info = json.loads((src / "meta" / "info.json").read_text())
    new_info["robot_type"] = "tong_dual_arm_rm75_ee_action"
    feats = new_info.setdefault("features", {})
    feats["action"] = _feat16()
    feats["action"]["fps"] = new_info.get("fps", 15)
    feats["action.joint"] = _feat_joint()
    feats["action.joint"]["fps"] = new_info.get("fps", 15)
    if limit is not None:
        new_info["total_episodes"] = int(limit)
        new_info["splits"] = {"train": f"0:{int(limit)}"}
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=4, ensure_ascii=False) + "\n")

    src_stats = src / "meta" / "stats.json"
    stats = json.loads(src_stats.read_text()) if src_stats.is_file() else {}
    if "action" in stats:
        stats["action.joint"] = stats["action"]
    stats["action"] = _stats(acc_action)
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")

    src_videos = src / "videos"
    if src_videos.exists() or src_videos.is_symlink():
        dst_videos = dst / "videos"
        if not dst_videos.exists() and not dst_videos.is_symlink():
            try:
                dst_videos.symlink_to(src_videos.resolve())
            except OSError:
                shutil.copytree(src_videos, dst_videos)

    action_all = np.concatenate(acc_action, axis=0)
    state_all = np.concatenate(acc_state, axis=0)
    meta = {
        "source": str(src.resolve()),
        "obs": "observation.state 16D joint 未改",
        "action": "16D LingBot EE [L xyz+quat | R xyz+quat | grips]",
        "action_names": ACTION_EE_NAMES,
        "fk": "复用源盘 action.ee（RealMan RM75 FK，绝对位姿）",
        "episodes": len(files),
        "frames": n_frames,
        "ee_xyz_action_mean": {
            "left": np.round(action_all[:, 0:3].mean(0), 4).tolist(),
            "right": np.round(action_all[:, 7:10].mean(0), 4).tolist(),
        },
        "quat_norm_mean": {
            "left": float(np.linalg.norm(action_all[:, 3:7], axis=1).mean()),
            "right": float(np.linalg.norm(action_all[:, 10:14], axis=1).mean()),
        },
        "state_joint_unchanged": True,
        "state_xyz_not_in_obs": True,
    }
    (dst / "meta" / "lingbot_ee_action.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )
    print("action xyz left", meta["ee_xyz_action_mean"]["left"], "right", meta["ee_xyz_action_mean"]["right"])
    print("quat norm", meta["quat_norm_mean"])
    print(f"写完 {len(files)} ep / {n_frames} frames → {dst}")
    return {
        "action_all": action_all,
        "state_all": state_all,
        "meta": meta,
        "info": new_info,
    }


def write_lingbot_norm(dst_json: Path, state_all: np.ndarray, action_all: np.ndarray, joint_norm: dict) -> None:
    """LingBot robot_config 用的 norm_stats：state 关节沿用旧文件，action 换成 EE。"""
    arm = joint_norm["norm_stats"]["observation.state.arm.position"]
    eff = joint_norm["norm_stats"]["observation.state.effector.position"]

    def pack(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(0).astype(np.float64).tolist(),
            "std": np.clip(arr.std(0), 1e-6, None).astype(np.float64).tolist(),
            "min": arr.min(0).astype(np.float64).tolist(),
            "max": arr.max(0).astype(np.float64).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).astype(np.float64).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).astype(np.float64).tolist(),
            "q02": np.quantile(arr, 0.02, axis=0).astype(np.float64).tolist(),
            "q98": np.quantile(arr, 0.98, axis=0).astype(np.float64).tolist(),
        }

    out = {
        "norm_stats": {
            "observation.state.arm.position": arm,
            "observation.state.effector.position": eff,
            "action.end.position": pack(action_all[:, 0:14]),
            "action.effector.position": pack(action_all[:, 14:16]),
        }
    }
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"norm → {dst_json}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="已有 action.ee 的 v2.1 盘（00073_rm75_ee_v21）")
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--norm-out", type=Path, default=None)
    p.add_argument("--joint-norm", type=Path, default=None, help="沿用 OBS 关节 norm 的 json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit-episodes", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = convert(args.src, args.dst, overwrite=args.overwrite, limit=args.limit_episodes)
    if args.norm_out:
        if not args.joint_norm or not args.joint_norm.is_file():
            raise FileNotFoundError("--norm-out 需要 --joint-norm")
        write_lingbot_norm(
            args.norm_out,
            result["state_all"],
            result["action_all"],
            json.loads(args.joint_norm.read_text()),
        )


if __name__ == "__main__":
    main()
