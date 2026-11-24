"""机器人 UDP 控制 I/O（infer/ 自包含，不依赖旧推理入口）。"""

from __future__ import annotations

import socket
import time
from typing import List, Tuple, Union

import numpy as np

from common.robot_dims import (
    LEFT_GRIPPER_DIM_INDEX,
    ORIGINAL_STATE_DIM,
    RIGHT_GRIPPER_DIM_INDEX,
)


def q_control_send(
    q_control: Union[np.ndarray, List[float]],
    sock: socket.socket,
    port_args: Tuple[str, int],
) -> None:
    try:
        payload = np.float64(q_control)
        sock.sendto(payload.tobytes(), port_args)
    except OSError as exc:
        raise RuntimeError(f"UDP 发送失败: {exc}") from exc


def pack_gripper_both_packet(left: float, right: float) -> np.ndarray:
    """同时设置左右夹爪开口比（new_exo_control_mode_lock set_gripper_position）。"""
    g = np.zeros(3, dtype=np.float64)
    g[1] = float(left)
    g[2] = float(right)
    return g


def pack_tongbot_control_packets(
    action_full_15d: np.ndarray,
    *,
    send_timestamp: float | None = None,
    gripper_udp_channel: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """16 维 full action → q_control(19) + gripper_control(3)。"""
    full = np.asarray(action_full_15d, dtype=np.float64).reshape(-1)
    if full.shape[0] != ORIGINAL_STATE_DIM:
        raise ValueError(f"action_full 期望 {ORIGINAL_STATE_DIM} 维，得到 {full.shape[0]}")

    channel = int(gripper_udp_channel)
    if channel not in (1, 2):
        raise ValueError(f"gripper_udp_channel 期望 1 或 2，得到 {channel}")

    q_control = np.zeros(19, dtype=np.float64)
    gripper_control = np.zeros(3, dtype=np.float64)
    q_control[0] = 2.0
    q_control[4:18] = full[0:14]
    q_control[18] = float(send_timestamp if send_timestamp is not None else time.time())
    gripper_control[1] = full[LEFT_GRIPPER_DIM_INDEX]
    gripper_control[2] = full[RIGHT_GRIPPER_DIM_INDEX]
    return q_control, gripper_control
