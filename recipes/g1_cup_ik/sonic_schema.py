"""GR00T N1.7 ``unitree_g1_sonic`` LeRobot v2.1 字段。

对齐 Isaac-GR00T ``embodiment_configs.py`` 的 UNITREE_G1_SONIC：
  video  ego_view 480x640
  state  43 关节（body-part 顺序）+ projected_gravity 3
  action 64 motion_token + 7 left_hand + 7 right_hand  （绝对，horizon 40）
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

FPS = 50
EGO_H = 480
EGO_W = 640
CHUNK_SIZE = 1000
TASK = "pick up the cup"

# PickPlaceToy / modality.json 的 43 关节名（不是 USD 交错顺序）
SONIC_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)

STATE_SLICES = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "left_hand": (22, 29),
    "right_arm": (29, 36),
    "right_hand": (36, 43),
    "projected_gravity": (43, 46),
}

# 43 里去掉双手 14 → 29 身体系，和 SONIC encoder 的 G1 mode 一致
BODY29_INDEX = list(range(0, 22)) + list(range(29, 36))
LEFT_HAND_INDEX = list(range(22, 29))
RIGHT_HAND_INDEX = list(range(36, 43))

STATE_DIM = 46  # 43 joints + 3 gravity
ACTION_DIM = 78  # 64 token + 7 + 7


def modality_json() -> dict:
    """GR00T 训练读的 meta/modality.json（concat 列 + start/end）。"""
    state = {k: {"start": s, "end": e} for k, (s, e) in STATE_SLICES.items()}
    return {
        "state": state,
        "action": {
            "motion_token": {"start": 0, "end": 64},
            "left_hand_joints": {"start": 64, "end": 71},
            "right_hand_joints": {"start": 71, "end": 78},
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
        },
        "annotation": {
            "human.task_description": {},
        },
    }


def _video_feature() -> dict:
    return {
        "dtype": "video",
        "shape": [EGO_H, EGO_W, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": EGO_H,
            "video.width": EGO_W,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def info_features() -> dict:
    action_names = (
        [f"motion_token_{i}" for i in range(64)]
        + [SONIC_JOINT_NAMES[i] for i in LEFT_HAND_INDEX]
        + [SONIC_JOINT_NAMES[i] for i in RIGHT_HAND_INDEX]
    )
    return {
        "observation.images.ego_view": _video_feature(),
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": list(SONIC_JOINT_NAMES) + ["gravity_x", "gravity_y", "gravity_z"],
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": action_names,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "annotation.human.task_description": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
    }


def empty_info(*, robot_type: str = "unitree_g1_sonic") -> dict:
    return {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": "0:0"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": info_features(),
        "discarded_episode_indices": [],
        "script_config": {
            "control_frequency": FPS,
            "embodiment": "UNITREE_G1_SONIC",
            "motion_token_source": "body29_fsq16_until_sonic_encoder",
            "task": TASK,
        },
    }


def remap_index(usd_joint_names: Sequence[str]) -> np.ndarray:
    """USD 关节序 → SONIC 43 序。缺失填 -1。"""
    lut = {n: i for i, n in enumerate(usd_joint_names)}
    idx = np.full(len(SONIC_JOINT_NAMES), -1, dtype=np.int64)
    for i, name in enumerate(SONIC_JOINT_NAMES):
        if name in lut:
            idx[i] = lut[name]
    return idx


def apply_remap(q_usd: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """q_usd: (..., J_usd) → (..., 43)。"""
    out = np.zeros(q_usd.shape[:-1] + (len(SONIC_JOINT_NAMES),), dtype=np.float32)
    valid = idx >= 0
    src = idx.copy()
    src[~valid] = 0
    gathered = np.take(q_usd, src, axis=-1)
    out[..., valid] = gathered[..., valid]
    return out


def apply_inverse_remap(q_sonic: np.ndarray, idx: np.ndarray, n_usd: int) -> np.ndarray:
    """q_sonic: (..., 43) → (..., n_usd)。没对齐上的 USD 关节保持 0。"""
    out = np.zeros(q_sonic.shape[:-1] + (n_usd,), dtype=np.float32)
    valid = idx >= 0
    for i, usd_i in enumerate(idx.tolist()):
        if usd_i >= 0:
            out[..., usd_i] = q_sonic[..., i]
    return out
