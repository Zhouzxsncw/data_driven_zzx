"""外骨骼关节 → robot_action 16D（与上位机 infer.exo.exo_kinematics 一致，下位机自包含）。"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

ORIGINAL_STATE_DIM = 16
LEFT_GRIPPER_DIM_INDEX = 14
RIGHT_GRIPPER_DIM_INDEX = 15


def _limit_torso_pos(torso_pos: float) -> float:
    return max(-20.0, min(float(torso_pos), 20.0))


def _limit_torso_pos_2(torso_pos: float) -> float:
    return max(-15.0, min(float(torso_pos), 20.0))


def set_range(angle_list: List[float]) -> List[float]:
    for i, angle in enumerate(angle_list):
        if i == 9 or i == 16:
            angle_list[i] = (float(angle) + 360.0) % 720.0 - 360.0
        else:
            angle_list[i] = (float(angle) + 180.0) % 360.0 - 180.0
        if i == 1:
            angle_list[i] = _limit_torso_pos_2(angle_list[i])
        elif i == 2:
            angle_list[i] = _limit_torso_pos(angle_list[i])
    return angle_list


def exo_joint_radians_to_degrees_17d(joint_rad: np.ndarray) -> List[float]:
    joints = [math.degrees(float(r)) for r in np.asarray(joint_rad, dtype=np.float64).reshape(-1)[:17]]
    if len(joints) != 17:
        raise ValueError(f"外骨骼关节角需要 17 维弧度，得到 {len(joints)}")
    joints[9] += 120.0
    joints[16] += 120.0
    return set_range(joints)


def infer_action_15d_from_tonggenesis_degrees_17d(
    degrees_17: Sequence[float],
    *,
    left_gripper: float,
    right_gripper: float | None = None,
) -> np.ndarray:
    deg = [float(x) for x in degrees_17[:17]]
    if len(deg) < 17:
        raise ValueError(f"degrees_17 需要 17 维，得到 {len(deg)}")
    rad_14: List[float] = []
    for i in range(14):
        cmd_idx = 3 + i
        joint_deg = deg[cmd_idx]
        if cmd_idx in (9, 16):
            joint_deg -= 120.0
        rad_14.append(math.radians(joint_deg))
    left = max(0.0, min(1.0, float(left_gripper)))
    right = max(0.0, min(1.0, float(right_gripper if right_gripper is not None else left_gripper)))
    out = np.asarray(rad_14 + [left, right], dtype=np.float64)
    if out.size != ORIGINAL_STATE_DIM:
        raise ValueError(f"action 期望 {ORIGINAL_STATE_DIM} 维，得到 {out.size}")
    return out


def exo_sample_to_robot_action_15d(
    joint_rad: np.ndarray,
    *,
    left_gripper: float,
    right_gripper: float | None = None,
) -> np.ndarray:
    degrees_17 = exo_joint_radians_to_degrees_17d(joint_rad)
    return infer_action_15d_from_tonggenesis_degrees_17d(
        degrees_17,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
    )
