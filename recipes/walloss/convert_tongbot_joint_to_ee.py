#!/usr/bin/env python3
"""TongBot LeRobot v2.1: keep 16D joint, add RM75 EE as extra columns.

Robot
-----
TongBot 双臂是 **RealMan / 睿尔曼 RM75**（7-DOF × 2 + 夹爪 × 2 = 16D）。
数据集 ``observation.state`` / ``action`` 存的是逻辑关节角（rad），
左右臂 J7 在落盘前已经 −120°；正解前必须加回 +120° 得到机身角（度）。

This script does **not** replace joint data. It writes a sibling dataset with:

- original ``observation.state`` / ``action``  (16D joint, unchanged)
- ``observation.state.ee`` / ``action.ee``     (20D absolute EE, WALL-OSS layout)

20D layout (absolute, each arm in **that arm's RM75 base frame**)::

    left_xyz(3) + left_rot6d(6) + left_gripper(1)
    + right_xyz(3) + right_rot6d(6) + right_gripper(1)

rot6d = top two rows of the ZYX rotation matrix, same as wall-x
``euler_to_matrix_zyx_6d_nb``. Poses are **absolute**; WALL-OSS loader turns
them into per-sample relative deltas when dof keys contain ``_relative``.

Videos are symlinked (not copied). FK uses RealMan ``Algo`` offline — no
hardware connection.

Example
-------
    PYTHONPATH=/home/xk/RM_API python convert_tongbot_joint_to_ee.py \\
        --src /home/xk/dataset/00086_20260827_RM_01_YJ_ZL_21 \\
        --dst /data/wall-oss/datasets/00086_rm75_ee
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

J7_OFFSET_DEG = 120.0
EE_DIM = 20
EE_NAMES = [
    "left_ee_x",
    "left_ee_y",
    "left_ee_z",
    "left_rot6d_0",
    "left_rot6d_1",
    "left_rot6d_2",
    "left_rot6d_3",
    "left_rot6d_4",
    "left_rot6d_5",
    "left_gripper",
    "right_ee_x",
    "right_ee_y",
    "right_ee_z",
    "right_rot6d_0",
    "right_rot6d_1",
    "right_rot6d_2",
    "right_rot6d_3",
    "right_rot6d_4",
    "right_rot6d_5",
    "right_gripper",
]


def _add_sdk_path(sdk_root: Path) -> None:
    sdk_root = sdk_root.expanduser().resolve()
    candidates = []
    if (sdk_root / "Robotic_Arm" / "rm_robot_interface.py").is_file():
        candidates.append(sdk_root)
    if sdk_root.name == "Robotic_Arm" and (sdk_root / "rm_robot_interface.py").is_file():
        candidates.append(sdk_root.parent)
    if not candidates:
        raise FileNotFoundError(
            f"找不到 RealMan SDK（需要 Robotic_Arm/rm_robot_interface.py）: {sdk_root}"
        )
    path = str(candidates[0])
    if path not in sys.path:
        sys.path.insert(0, path)


def _make_algo(sdk_root: Path, force_type_name: str):
    _add_sdk_path(sdk_root)
    from Robotic_Arm.rm_robot_interface import (  # noqa: WPS433
        Algo,
        rm_force_type_e,
        rm_robot_arm_model_e,
    )

    force = getattr(rm_force_type_e, force_type_name)
    algo = Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E, force)
    if int(algo.arm_dof) != 7:
        raise RuntimeError(f"RM75 Algo dof={algo.arm_dof}, expected 7")
    return algo


def logical_rad_to_physical_deg(joint7: np.ndarray, j7_offset_deg: float) -> np.ndarray:
    """Dataset logical rad (J7 already −offset) → SDK physical degrees."""
    deg = np.rad2deg(np.asarray(joint7, dtype=np.float64))
    deg = np.atleast_2d(deg).copy()
    deg[:, 6] += float(j7_offset_deg)
    return deg


def euler_zyx_to_rot6d(euler_xyz: np.ndarray) -> np.ndarray:
    """Match wall-x ``euler_to_matrix_zyx_6d_nb``: top two rows of ZYX R."""
    e = np.asarray(euler_xyz, dtype=np.float64)
    single = e.ndim == 1
    if single:
        e = e[np.newaxis, :]
    roll, pitch, yaw = e[:, 0], e[:, 1], e[:, 2]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    out = np.stack([r00, r01, r02, r10, r11, r12], axis=1)
    return out[0] if single else out


def fk_pose_euler(algo, joint7_physical_deg: np.ndarray) -> np.ndarray:
    joints = [float(x) for x in np.asarray(joint7_physical_deg, dtype=np.float64).reshape(-1)]
    if len(joints) != 7:
        raise ValueError(f"需要 7 维物理关节角（度），得到 {len(joints)}")
    pose = algo.rm_algo_forward_kinematics(joints, 1)
    return np.asarray(pose, dtype=np.float64)


def joints16_to_ee20(
    algo,
    joints16: np.ndarray,
    j7_offset_deg: float,
) -> np.ndarray:
    """(N, 16) logical rad+gripper → (N, 20) absolute WALL-OSS EE."""
    arr = np.asarray(joints16, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[1] < 16:
        raise ValueError(f"期望至少 16 维 [左7|右7|左爪|右爪]，得到 {arr.shape}")
    n = arr.shape[0]
    out = np.empty((n, EE_DIM), dtype=np.float32)
    left_deg = logical_rad_to_physical_deg(arr[:, 0:7], j7_offset_deg)
    right_deg = logical_rad_to_physical_deg(arr[:, 7:14], j7_offset_deg)
    for i in range(n):
        left_pose = fk_pose_euler(algo, left_deg[i])
        right_pose = fk_pose_euler(algo, right_deg[i])
        out[i, 0:3] = left_pose[0:3]
        out[i, 3:9] = euler_zyx_to_rot6d(left_pose[3:6])
        out[i, 9] = arr[i, 14]
        out[i, 10:13] = right_pose[0:3]
        out[i, 13:19] = euler_zyx_to_rot6d(right_pose[3:6])
        out[i, 19] = arr[i, 15]
    return out


def _ee_feature() -> dict:
    return {
        "dtype": "float32",
        "shape": [EE_DIM],
        "names": list(EE_NAMES),
        "fps": 15,
    }


def _stack_object_column(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in series.to_numpy()], axis=0)


def _as_object_rows(arr: np.ndarray) -> list[np.ndarray]:
    return [row.astype(np.float32) for row in arr]


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _running_stats(acc: dict[str, list[np.ndarray]], key: str, arr: np.ndarray) -> None:
    acc.setdefault(key, []).append(np.asarray(arr, dtype=np.float64))


def _finalize_stats(chunks: list[np.ndarray]) -> dict:
    data = np.concatenate(chunks, axis=0)
    return {
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
        "count": [int(data.shape[0])] * data.shape[1],
    }


_ALGO = None


def _init_worker(sdk_root: str, force_type: str) -> None:
    global _ALGO
    _ALGO = _make_algo(Path(sdk_root), force_type)


def _convert_one(job: tuple) -> tuple[str, np.ndarray, np.ndarray]:
    """One episode: FK both arms, write parquet, return EE arrays for stats."""
    pq_s, src_s, dst_s, j7 = job
    if _ALGO is None:
        raise RuntimeError("worker Algo 未初始化")
    pq = Path(pq_s)
    src = Path(src_s)
    dst = Path(dst_s)
    rel = pq.relative_to(src)
    out_pq = dst / rel
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(pq)
    state = _stack_object_column(df["observation.state"])
    action = _stack_object_column(df["action"])
    state_ee = joints16_to_ee20(_ALGO, state, j7)
    action_ee = joints16_to_ee20(_ALGO, action, j7)
    df["observation.state.ee"] = _as_object_rows(state_ee)
    df["action.ee"] = _as_object_rows(action_ee)
    df.to_parquet(out_pq, index=False)
    return str(rel), state_ee, action_ee


def convert_dataset(
    src: Path,
    dst: Path,
    *,
    sdk_root: Path,
    force_type: str,
    j7_offset_deg: float,
    limit_episodes: int | None,
    overwrite: bool,
    workers: int = 1,
) -> None:
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") not in {"v2.0", "v2.1"}:
        print(f"警告: codebase_version={info.get('codebase_version')}，脚本按 v2.1 parquet 写")

    parquet_files = sorted((src / "data").rglob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"没有 parquet: {src / 'data'}")
    if limit_episodes is not None:
        parquet_files = parquet_files[: int(limit_episodes)]

    if dst.exists() and not overwrite:
        existing = list((dst / "data").rglob("episode_*.parquet")) if (dst / "data").exists() else []
        if existing:
            raise FileExistsError(f"{dst} 已有 {len(existing)} 个 parquet，加 --overwrite 才覆盖")

    dst.mkdir(parents=True, exist_ok=True)
    stats_acc: dict[str, list[np.ndarray]] = {}
    n_workers = max(1, int(workers))
    jobs = [
        (str(pq), str(src), str(dst), float(j7_offset_deg)) for pq in parquet_files
    ]

    if n_workers == 1:
        _init_worker(str(sdk_root), force_type)
        results = [_convert_one(job) for job in tqdm(jobs, desc="FK joint→EE")]
    else:
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        print(f"FK workers={n_workers} episodes={len(jobs)}", flush=True)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(str(sdk_root), force_type),
        ) as pool:
            results = list(tqdm(pool.map(_convert_one, jobs), total=len(jobs), desc="FK joint→EE"))

    for _rel, state_ee, action_ee in results:
        _running_stats(stats_acc, "observation.state.ee", state_ee)
        _running_stats(stats_acc, "action.ee", action_ee)

    (dst / "meta").mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl", "modality.json"):
        src_meta = src / "meta" / name
        if src_meta.is_file():
            shutil.copy2(src_meta, dst / "meta" / name)

    new_info = json.loads(info_path.read_text())
    new_info["robot_type"] = "tong_dual_arm_rm75_ee"
    feats = new_info.setdefault("features", {})
    feats["observation.state.ee"] = _ee_feature()
    feats["action.ee"] = _ee_feature()
    feats["observation.state.ee"]["fps"] = new_info.get("fps", 15)
    feats["action.ee"]["fps"] = new_info.get("fps", 15)
    if limit_episodes is not None:
        new_info["total_episodes"] = int(limit_episodes)
        new_info["splits"] = {"train": f"0:{int(limit_episodes)}"}
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=4, ensure_ascii=False) + "\n")

    src_stats = src / "meta" / "stats.json"
    stats = json.loads(src_stats.read_text()) if src_stats.is_file() else {}
    for key, chunks in stats_acc.items():
        stats[key] = _finalize_stats(chunks)
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")

    src_videos = src / "videos"
    if src_videos.is_dir():
        _symlink_or_copy(src_videos, dst / "videos")

    conversion_meta = {
        "source": str(src.resolve()),
        "robot": "RealMan RM75 dual-arm",
        "sdk": str(sdk_root),
        "force_type": force_type,
        "j7_offset_deg": j7_offset_deg,
        "workers": int(n_workers),
        "ee_layout": EE_NAMES,
        "note": (
            "observation.state / action 仍是 16D joint；"
            "observation.state.ee / action.ee 是各臂基座系下的绝对 EE。"
            "WALL-OSS 训练请把 key_mappings.state/action 指到 *.ee，"
            "dof 用 follow_*_ee_*_relative，由 loader 做相对增量。"
        ),
    }
    (dst / "meta" / "joint_to_ee.json").write_text(
        json.dumps(conversion_meta, indent=2, ensure_ascii=False) + "\n"
    )

    state_ee_all = np.concatenate(stats_acc["observation.state.ee"], axis=0)
    print(
        "EE xyz left  mean",
        np.round(state_ee_all[:, 0:3].mean(0), 4),
        "std",
        np.round(state_ee_all[:, 0:3].std(0), 4),
    )
    print(
        "EE xyz right mean",
        np.round(state_ee_all[:, 10:13].mean(0), 4),
        "std",
        np.round(state_ee_all[:, 10:13].std(0), 4),
    )
    print(f"写完 {len(parquet_files)} episodes → {dst}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="原始 LeRobot v2.1 根目录（含 data/ meta/ videos/）")
    p.add_argument("--dst", type=Path, required=True, help="输出根目录（新建；videos 默认 symlink）")
    p.add_argument("--sdk-root", type=Path, default=Path("/home/xk/RM_API"), help="RealMan Python SDK 根（含 Robotic_Arm/）")
    p.add_argument(
        "--force-type",
        default="RM_MODEL_RM_B_E",
        help="rm_force_type_e 名称，标准末端 RM75-B 用 RM_MODEL_RM_B_E",
    )
    p.add_argument("--j7-offset-deg", type=float, default=J7_OFFSET_DEG)
    p.add_argument("--limit-episodes", type=int, default=None, help="只转前 N 条，用于冒烟")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="episode 级进程数。1 为单进程；>1 用 spawn，每个进程自建 RM Algo",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(
        args.src,
        args.dst,
        sdk_root=args.sdk_root,
        force_type=args.force_type,
        j7_offset_deg=args.j7_offset_deg,
        limit_episodes=args.limit_episodes,
        overwrite=args.overwrite,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
