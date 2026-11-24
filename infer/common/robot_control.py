"""TongBot 下位机 UDP 控制包打包（legacy / native 共用）。"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from common.robot_dims import (
    LEFT_GRIPPER_DIM_INDEX,
    ORIGINAL_STATE_DIM,
    RIGHT_GRIPPER_DIM_INDEX,
)


def pack_tongbot_control_packets(
    action_full_15d: np.ndarray,
    *,
    send_timestamp: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 16 维 full action 打包为 q_control(19) + gripper_control(3)。

    与 pi05_policy_infer / policy_infer_h_new_4 一致：
      q_control[0] = 2
      q_control[4:18] = action_full[0:14]   # 14 关节
      q_control[18] = send_timestamp
      gripper_control[1] = action_full[14]   # 左夹爪
      gripper_control[2] = action_full[15]   # 右夹爪
    """
    full = np.asarray(action_full_15d, dtype=np.float64).reshape(-1)
    if full.shape[0] != ORIGINAL_STATE_DIM:
        raise ValueError(f"action_full 期望 {ORIGINAL_STATE_DIM} 维，得到 {full.shape[0]}")

    q_control = np.zeros(19, dtype=np.float64)
    gripper_control = np.zeros(3, dtype=np.float64)
    q_control[0] = 2.0
    q_control[1 + 3 : 15 + 3] = full[0:14]
    q_control[18] = float(send_timestamp if send_timestamp is not None else time.time())
    gripper_control[1] = full[LEFT_GRIPPER_DIM_INDEX]
    gripper_control[2] = full[RIGHT_GRIPPER_DIM_INDEX]
    return q_control, gripper_control
