"""标准初始位姿常量（与 new_exo_control_mode_lock.py 保持一致）。"""

from __future__ import annotations

from typing import List

# 17 维关节角（度）：腰 3 + 左臂 7 + 右臂 7
STANDARD_INIT_JOINT_17D_DEG: List[float] = [
    0.0,
    2.5,
    0.0,
    -103.0,
    -73.67,
    89.0,
    -8.31,
    -22.0,
    -75.5,
    76.4,
    9.8,
    82.1,
    -14.9,
    12.1,
    2.59,
    -29.4,
    200.6,
]
