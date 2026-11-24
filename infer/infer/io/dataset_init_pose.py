"""从 LeRobot 数据集 episode 首帧统计推理初始位姿。"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from infer.config_loader import InferConfig
from infer.io.direct_robot_io import infer_state_15d_to_joint_17d_degrees
from infer.io.robot_init_pose import STANDARD_INIT_JOINT_17D_DEG
from common.robot_dims import (
    LEFT_GRIPPER_DIM_INDEX,
    ORIGINAL_STATE_DIM,
    RIGHT_GRIPPER_DIM_INDEX,
)

LOGGER = logging.getLogger(__name__)

LEFT_GRIPPER_15D_INDEX = LEFT_GRIPPER_DIM_INDEX
RIGHT_GRIPPER_15D_INDEX = RIGHT_GRIPPER_DIM_INDEX


@dataclass(frozen=True)
class DatasetInitPoseResult:
    joint_17d_deg: List[float]
    gripper_position: List[float]
    num_episodes: int
    state_dim: int
    aggregation: str
    native_median: List[float]


def load_state_feature_names(dataset_root: Path) -> List[str]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return []
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feat = (info.get("features") or {}).get("observation.state") or {}
    names = feat.get("names")
    shape = feat.get("shape") or []
    dim = int(shape[0]) if shape else 0
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    return [f"dim{i}" for i in range(dim)]


def collect_episode_first_states(dataset_root: Path) -> Tuple[np.ndarray, int]:
    """返回 (num_episodes, state_dim) 的各 episode 首帧 observation.state。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "从数据集统计初始位姿需要 pyarrow：pip install pyarrow"
        ) from exc

    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据集缺少 data/ 目录: {data_dir}")

    by_episode: dict[int, tuple[int, np.ndarray]] = {}
    for pq_path in sorted(data_dir.glob("**/*.parquet")):
        table = pq.read_table(
            pq_path,
            columns=["episode_index", "frame_index", "observation.state"],
        )
        episodes = table.column("episode_index").to_pylist()
        frames = table.column("frame_index").to_pylist()
        states = table.column("observation.state").to_pylist()
        for ep, frame_idx, state in zip(episodes, frames, states):
            ep_i = int(ep)
            fi = int(frame_idx)
            arr = np.asarray(state, dtype=np.float64).reshape(-1)
            prev = by_episode.get(ep_i)
            if prev is None or fi < prev[0]:
                by_episode[ep_i] = (fi, arr)

    if not by_episode:
        raise ValueError(f"未在 {data_dir} 找到任何 episode 首帧")

    stacked = np.stack([item[1] for _, item in sorted(by_episode.items())], axis=0)
    return stacked, len(by_episode)


def _aggregate_states(states: np.ndarray, aggregation: str) -> np.ndarray:
    if aggregation == "mean":
        return np.mean(states, axis=0)
    if aggregation in ("median", "dataset_median"):
        return np.median(states, axis=0)
    raise ValueError(f"未知 aggregation: {aggregation}")


def _looks_like_degrees(values: Sequence[float]) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return False
    return float(np.max(np.abs(arr))) > 2.0 * math.pi + 0.01


def native_state_to_infer_15d(
    state_native: np.ndarray,
    *,
    keep_dims: Sequence[int],
) -> np.ndarray:
    """数据集原生 state → infer 16D（关节弧度 + 左右夹爪 0~1）。"""
    state = np.asarray(state_native, dtype=np.float64).reshape(-1)
    full = np.full(ORIGINAL_STATE_DIM, np.nan, dtype=np.float64)

    if state.size == len(keep_dims):
        for i, dim in enumerate(keep_dims):
            dim_i = int(dim)
            val = float(state[i])
            if dim_i < 14 and _looks_like_degrees([val]):
                val = math.radians(val)
            full[dim_i] = val
        return full

    if state.size >= ORIGINAL_STATE_DIM:
        for i in range(ORIGINAL_STATE_DIM):
            val = float(state[i])
            if i < 14 and _looks_like_degrees([val]):
                val = math.radians(val)
            full[i] = val
        return full

    raise ValueError(
        f"无法映射 observation.state 维数 {state.size} 到 infer 16D "
        f"(keep_dims 数量={len(keep_dims)})"
    )


def merge_dataset_pose_with_standard(
    infer_15d: np.ndarray,
    standard_17d: Sequence[float],
) -> Tuple[List[float], List[float]]:
    """用数据集覆盖 standard 中有效的臂/夹爪 DOF，其余保留 standard。"""
    standard = [float(x) for x in standard_17d]
    if len(standard) != 17:
        raise ValueError(f"standard_init_joint_degrees 需要 17 维，得到 {len(standard)}")

    derived_17d, _gripper = infer_state_15d_to_joint_17d_degrees(infer_15d)
    pose = list(standard)

    if not np.isnan(infer_15d[0:7]).all():
        pose[3:10] = derived_17d[3:10]
    if not np.isnan(infer_15d[7:14]).all():
        pose[10:17] = derived_17d[10:17]

    left_g = (
        float(infer_15d[LEFT_GRIPPER_15D_INDEX])
        if not np.isnan(infer_15d[LEFT_GRIPPER_15D_INDEX])
        else None
    )
    right_g = (
        float(infer_15d[RIGHT_GRIPPER_15D_INDEX])
        if not np.isnan(infer_15d[RIGHT_GRIPPER_15D_INDEX])
        else left_g
    )
    if left_g is not None or right_g is not None:
        left_g = float(np.clip(left_g if left_g is not None else 0.7, 0.0, 1.0))
        right_g = float(np.clip(right_g if right_g is not None else left_g, 0.0, 1.0))
        gripper_pos = [left_g, right_g]
    else:
        gripper_pos = [0.7, 0.7]

    return pose, gripper_pos


def analyze_dataset_init_pose(
    dataset_root: Path,
    *,
    keep_dims: Sequence[int],
    aggregation: str = "median",
    standard_17d: Optional[Sequence[float]] = None,
) -> DatasetInitPoseResult:
    root = Path(dataset_root).expanduser().resolve()
    states, num_eps = collect_episode_first_states(root)
    agg_key = "median" if aggregation.endswith("median") else "mean"
    native = _aggregate_states(states, agg_key)
    infer_15d = native_state_to_infer_15d(native, keep_dims=keep_dims)
    standard = list(standard_17d or STANDARD_INIT_JOINT_17D_DEG)
    joint_17d, gripper = merge_dataset_pose_with_standard(infer_15d, standard)

    names = load_state_feature_names(root)
    LOGGER.info(
        "数据集初始位姿: episodes=%d aggregation=%s state_dim=%d names=%s",
        num_eps,
        agg_key,
        native.size,
        names[:8] if names else "?",
    )
    LOGGER.info(
        "数据集首帧 median/mean → joint_17d(度) 左臂=%s 右臂=%s gripper=%s",
        [round(v, 2) for v in joint_17d[3:10]],
        [round(v, 2) for v in joint_17d[10:17]],
        [round(v, 3) for v in gripper],
    )

    return DatasetInitPoseResult(
        joint_17d_deg=joint_17d,
        gripper_position=gripper,
        num_episodes=num_eps,
        state_dim=int(native.size),
        aggregation=agg_key,
        native_median=[float(x) for x in native.tolist()],
    )


def resolve_inference_init_pose(cfg: InferConfig) -> Tuple[List[float], List[float]]:
    """解析组间 reset 使用的 17D 关节角（度）与夹爪初值。"""
    dr = cfg.direct_robot
    standard = dr.standard_init_joint_degrees or STANDARD_INIT_JOINT_17D_DEG
    source = str(getattr(dr, "init_pose_source", "standard") or "standard").lower()

    if source == "standard":
        return [float(x) for x in standard], list(dr.gripper_initial_position or [0.7, 0.7])

    if not cfg.policy.dataset_root:
        LOGGER.warning("init_pose_source=%s 但未配置 dataset_root，回退 standard_init", source)
        return [float(x) for x in standard], list(dr.gripper_initial_position or [0.7, 0.7])

    try:
        result = analyze_dataset_init_pose(
            cfg.policy.dataset_root,
            keep_dims=list(cfg.policy.keep_dims),
            aggregation=source,
            standard_17d=standard,
        )
    except Exception as exc:
        LOGGER.warning("数据集初始位姿分析失败，回退 standard_init: %s", exc)
        return [float(x) for x in standard], list(dr.gripper_initial_position or [0.7, 0.7])

    return result.joint_17d_deg, result.gripper_position
