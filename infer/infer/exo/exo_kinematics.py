"""外骨骼关节角 → direct_robot 17D 度命令（与 new_exo_control_mode_lock 一致）。"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from common.robot_dims import ORIGINAL_STATE_DIM
from infer.io.direct_robot_io import action_15d_to_tonggenesis_degrees, set_range


def exo_joint_radians_to_degrees_17d(joint_rad: np.ndarray) -> List[float]:
    """
    外骨骼 read() 返回的 17 关节弧度 → TongGenesis 17D 度命令。

    与下位机 robot_control_listener 一致：转度、关节 9/16 加 120°、set_range。
    """
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
    """17D TongGenesis 度命令 → infer/data_collect 共用的 16D action（弧度 + 左右夹爪）。

    与 action_15d_to_tonggenesis_degrees / exo_joint_radians_to_degrees_17d 互逆。
    """
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
    """外骨骼样本 → data_collect robot_action 包同布局的 16D（弧度）。

    夹爪维（index 14/15）仅来自外骨骼 UDP，不得混入下位机 Dynamixel 回传或下发缓存。
    """
    degrees_17 = exo_joint_radians_to_degrees_17d(joint_rad)
    return infer_action_15d_from_tonggenesis_degrees_17d(
        degrees_17,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
    )


def robot_state_15d_to_exo_joint_degrees_17(
    state_15d: np.ndarray,
    *,
    hold_waist_deg: Sequence[float] | None = None,
) -> List[float]:
    """机械臂观测 16D → 外骨骼舵机目标角（度，17 维，与 present_angle_read 同单位）。

    布局（与 TongGenesis / exo dxl 0..16 一致）::

        [0:3]  腰
        [3:10] 左臂 7
        [10:17] 右臂 7

    15D 不含腰；默认若未提供 ``hold_waist_deg``，腰目标为 0。
    自动对齐应传入当前外骨骼腰角，避免把腰拧到 0。
    关节 9/16（左右 J7）去掉 TongGenesis 命令空间的 +120° 偏移。
    """
    command = action_15d_to_tonggenesis_degrees(
        np.asarray(state_15d, dtype=np.float64).reshape(-1)
    )
    deg = [float(x) for x in command[:17]]
    deg[9] -= 120.0
    deg[16] -= 120.0
    # wrap to [-180, 180] 与 TGExo._limit_angle / present_angle 一致
    out: List[float] = []
    for a in deg:
        out.append((float(a) + 180.0) % 360.0 - 180.0)
    if hold_waist_deg is not None:
        waist = list(hold_waist_deg)[:3]
        if len(waist) < 3:
            raise ValueError(f"hold_waist_deg 需要 ≥3 维，得到 {len(waist)}")
        for i in range(3):
            out[i] = (float(waist[i]) + 180.0) % 360.0 - 180.0
    return out
