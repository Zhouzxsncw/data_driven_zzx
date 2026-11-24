"""robot_state / robot_action 16D 包中的夹爪维处理（与 data_collect modalities 一致）。"""

from __future__ import annotations

import numpy as np

from common.gripper_udp_packet import clamp_gripper_ratio

LEFT_GRIPPER_DIM_INDEX = 14
RIGHT_GRIPPER_DIM_INDEX = 15
# 兼容旧引用
GRIPPER_DIM_INDEX = LEFT_GRIPPER_DIM_INDEX
ROBOT_PACKET_DIM = 16


def apply_left_gripper_to_robot_packet_15d(
    packet_15d: np.ndarray,
    left_ratio: float,
) -> np.ndarray:
    """将 16D 包第 14 维设为左夹爪开口比（state 用下位机 UDP 实测）。"""
    out = np.asarray(packet_15d, dtype=np.float64).reshape(-1).copy()
    if out.size < ROBOT_PACKET_DIM:
        raise ValueError(f"robot 包期望至少 {ROBOT_PACKET_DIM} 维，得到 {out.size}")
    out[LEFT_GRIPPER_DIM_INDEX] = clamp_gripper_ratio(left_ratio)
    return out


def apply_right_gripper_to_robot_packet(
    packet: np.ndarray,
    right_ratio: float,
) -> np.ndarray:
    """将 16D 包第 15 维设为右夹爪开口比。"""
    out = np.asarray(packet, dtype=np.float64).reshape(-1).copy()
    if out.size < ROBOT_PACKET_DIM:
        raise ValueError(f"robot 包期望至少 {ROBOT_PACKET_DIM} 维，得到 {out.size}")
    out[RIGHT_GRIPPER_DIM_INDEX] = clamp_gripper_ratio(right_ratio)
    return out


def apply_grippers_to_robot_packet(
    packet: np.ndarray,
    *,
    left_ratio: float | None = None,
    right_ratio: float | None = None,
) -> np.ndarray:
    """按需写入左右夹爪维。"""
    out = np.asarray(packet, dtype=np.float64).reshape(-1).copy()
    if left_ratio is not None:
        out = apply_left_gripper_to_robot_packet_15d(out, left_ratio)
    if right_ratio is not None:
        out = apply_right_gripper_to_robot_packet(out, right_ratio)
    return out
