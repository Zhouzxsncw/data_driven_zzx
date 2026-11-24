"""GR00T N1.7 RTC 的动作轴速度掩码。

与 Isaac-GR00T ``get_action_with_features`` 的 ``vel_strength`` 同一公式：
前 ``frozen_steps`` 帧为 0，重叠段是归一化指数斜坡，之后为 1。
π0.5 去噪把它乘在速度上，时间方向仍用 π0.5 自己的负 dt。
"""

from __future__ import annotations

import numpy as np


def resolve_gr00t_rtc_window(
    horizon: int,
    prefix_horizon: int,
    prev_len: int,
    inference_delay: int,
    overlap_steps: int | None,
    frozen_steps: int | None,
) -> tuple[int, int, int]:
    """有效重叠长度不超过上一包还没执行的步，避免把右侧补零抄进噪声。

    返回 ``(valid, overlap, frozen)``。上一包在 policy 层已经切掉已执行前缀并右侧补零，
    所以要写进新噪声的是左前缀，不是补零后的尾巴。
    """
    valid = max(0, min(int(prefix_horizon), int(horizon), int(prev_len)))
    overlap = valid if overlap_steps is None else int(overlap_steps)
    overlap = max(0, min(overlap, valid))
    frozen = int(inference_delay) if frozen_steps is None else int(frozen_steps)
    frozen = max(0, min(frozen, overlap))
    return valid, overlap, frozen


def gr00t_rtc_vel_strength(
    horizon: int,
    frozen_steps: int,
    overlap_steps: int,
    ramp_rate: float,
) -> np.ndarray:
    """返回 float32 向量，长度 ``horizon``。"""
    length = int(horizon)
    if length < 1:
        raise ValueError(f"horizon 必须 >= 1，得到 {horizon}")
    frozen = max(0, min(int(frozen_steps), length))
    overlap = max(frozen, min(int(overlap_steps), length))
    strength = np.ones(length, dtype=np.float32)
    if frozen > 0:
        strength[:frozen] = 0
    intermediate = overlap - frozen
    if intermediate > 0:
        grid = np.linspace(0.0, 1.0, intermediate + 2, dtype=np.float64)
        ramp = 1.0 - np.exp(-float(ramp_rate) * grid)
        ramp = ramp / max(float(ramp[-1]), 1e-8)
        strength[frozen:overlap] = ramp[1:-1].astype(np.float32)
    return strength
