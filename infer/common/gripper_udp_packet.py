"""下位机夹爪状态 UDP 包（30016）。

布局（float64）:
  [0] left_ratio
  [1] right_ratio
  [2] timestamp_sec
  [3] left_current_mA   （可选；旧包无此维）
  [4] right_current_mA  （可选；旧包无此维）

兼容：长度 >= 3 可解析；电流缺失或 NaN → None。
"""

from __future__ import annotations

import math
import time
from typing import NamedTuple, Optional, Tuple

import numpy as np

GRIPPER_STATE_PACKET_DIM_MIN = 3
GRIPPER_STATE_PACKET_DIM = 5
DEFAULT_GRIPPER_STATE_PORT = 30016


def clamp_gripper_ratio(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


class GripperStatePacket(NamedTuple):
    left: float
    right: float
    timestamp: float
    left_current_mA: Optional[float] = None
    right_current_mA: Optional[float] = None


def pack_gripper_state_packet(
    left: float,
    right: float,
    *,
    timestamp: float | None = None,
    left_current_mA: float | None = None,
    right_current_mA: float | None = None,
) -> bytes:
    """打包夹爪状态；电流未知时写 NaN。"""
    ts = time.time() if timestamp is None else float(timestamp)
    left_i = (
        float("nan")
        if left_current_mA is None or not math.isfinite(float(left_current_mA))
        else float(left_current_mA)
    )
    right_i = (
        float("nan")
        if right_current_mA is None or not math.isfinite(float(right_current_mA))
        else float(right_current_mA)
    )
    arr = np.asarray(
        [
            clamp_gripper_ratio(left),
            clamp_gripper_ratio(right),
            ts,
            left_i,
            right_i,
        ],
        dtype=np.float64,
    )
    return arr.tobytes()


def parse_gripper_state_packet(data: bytes) -> GripperStatePacket:
    """解析夹爪状态 UDP 包（兼容旧 3 维包）。"""
    arr = np.frombuffer(data, dtype=np.float64)
    if arr.size < GRIPPER_STATE_PACKET_DIM_MIN:
        raise ValueError(
            f"夹爪状态包长度不足: 期望 >= {GRIPPER_STATE_PACKET_DIM_MIN} 个 float64，得到 {arr.size}"
        )
    left_i = _optional_float(arr[3]) if arr.size >= 5 else None
    right_i = _optional_float(arr[4]) if arr.size >= 5 else None
    return GripperStatePacket(
        left=clamp_gripper_ratio(float(arr[0])),
        right=clamp_gripper_ratio(float(arr[1])),
        timestamp=float(arr[2]),
        left_current_mA=left_i,
        right_current_mA=right_i,
    )


# 旧调用方兼容：仍可解包三元
def parse_gripper_state_packet_legacy(data: bytes) -> Tuple[float, float, float]:
    pkt = parse_gripper_state_packet(data)
    return pkt.left, pkt.right, pkt.timestamp
