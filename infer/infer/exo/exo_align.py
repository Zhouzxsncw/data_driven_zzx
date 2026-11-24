"""外骨骼对齐门控：关节空间误差（与 canfd 发令空间一致）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from common.robot_dims import LEFT_ARM_DIMS, ORIGINAL_STATE_DIM, RIGHT_ARM_DIMS


@dataclass(frozen=True)
class ExoAlignStatus:
    """分臂关节对齐状态；夹爪不参与门控。"""

    ok: bool
    has_exo: bool
    has_robot: bool
    left_max_deg: float
    left_rms_deg: float
    right_max_deg: float
    right_rms_deg: float
    max_thresh_deg: float
    rms_thresh_deg: float
    check_left: bool = True
    check_right: bool = True

    @property
    def summary(self) -> str:
        if not self.has_exo:
            return "等待外骨骼状态包…"
        if not self.has_robot:
            return "等待机械臂状态…"
        flag = "OK" if self.ok else "未对齐"
        parts = [flag]
        if self.check_left:
            parts.append(
                f"L max={self.left_max_deg:.1f}° rms={self.left_rms_deg:.1f}°"
            )
        if self.check_right:
            parts.append(
                f"R max={self.right_max_deg:.1f}° rms={self.right_rms_deg:.1f}°"
            )
        parts.append(
            f"阈值 max≤{self.max_thresh_deg:.1f}° rms≤{self.rms_thresh_deg:.1f}°"
        )
        return " | ".join(parts)


def _arm_errors_deg(
    exo_rad: np.ndarray,
    robot_rad: np.ndarray,
    dims: Sequence[int],
) -> tuple[float, float]:
    diffs = []
    for i in dims:
        d = float(exo_rad[i] - robot_rad[i])
        # wrap to [-pi, pi] then degrees
        d = (d + math.pi) % (2.0 * math.pi) - math.pi
        diffs.append(abs(math.degrees(d)))
    if not diffs:
        return 0.0, 0.0
    arr = np.asarray(diffs, dtype=np.float64)
    return float(arr.max()), float(np.sqrt(np.mean(arr * arr)))


def arm_sides_from_keep_dims(keep_dims: Sequence[int]) -> tuple[bool, bool]:
    """根据 policy.keep_dims 决定对齐检查哪些臂。"""
    dims = {int(d) for d in keep_dims}
    check_left = any(d in dims for d in LEFT_ARM_DIMS)
    check_right = any(d in dims for d in RIGHT_ARM_DIMS)
    if not check_left and not check_right:
        # 异常配置：回退双臂，避免门控恒真
        return True, True
    return check_left, check_right


def compute_exo_align_status(
    exo_action_15d: Optional[np.ndarray],
    robot_state_15d: Optional[np.ndarray],
    *,
    max_joint_err_deg: float,
    rms_joint_err_deg: float,
    check_left: bool = True,
    check_right: bool = True,
) -> ExoAlignStatus:
    """比较 exo/robot 臂关节（15D/16D 布局）；可按单/双臂裁剪。"""
    max_th = float(max_joint_err_deg)
    rms_th = float(rms_joint_err_deg)
    if not check_left and not check_right:
        check_left, check_right = True, True
    if exo_action_15d is None:
        return ExoAlignStatus(
            ok=False,
            has_exo=False,
            has_robot=robot_state_15d is not None,
            left_max_deg=float("nan"),
            left_rms_deg=float("nan"),
            right_max_deg=float("nan"),
            right_rms_deg=float("nan"),
            max_thresh_deg=max_th,
            rms_thresh_deg=rms_th,
            check_left=check_left,
            check_right=check_right,
        )
    if robot_state_15d is None:
        return ExoAlignStatus(
            ok=False,
            has_exo=True,
            has_robot=False,
            left_max_deg=float("nan"),
            left_rms_deg=float("nan"),
            right_max_deg=float("nan"),
            right_rms_deg=float("nan"),
            max_thresh_deg=max_th,
            rms_thresh_deg=rms_th,
            check_left=check_left,
            check_right=check_right,
        )

    exo = np.asarray(exo_action_15d, dtype=np.float64).reshape(-1)
    robot = np.asarray(robot_state_15d, dtype=np.float64).reshape(-1)
    if exo.size < ORIGINAL_STATE_DIM or robot.size < ORIGINAL_STATE_DIM:
        raise ValueError(
            f"对齐需要 ≥{ORIGINAL_STATE_DIM} 维，得到 exo={exo.size} robot={robot.size}"
        )

    left_max = left_rms = 0.0
    right_max = right_rms = 0.0
    ok = True
    if check_left:
        left_max, left_rms = _arm_errors_deg(exo, robot, LEFT_ARM_DIMS)
        ok = ok and left_max <= max_th and left_rms <= rms_th
    if check_right:
        right_max, right_rms = _arm_errors_deg(exo, robot, RIGHT_ARM_DIMS)
        ok = ok and right_max <= max_th and right_rms <= rms_th
    return ExoAlignStatus(
        ok=ok,
        has_exo=True,
        has_robot=True,
        left_max_deg=left_max,
        left_rms_deg=left_rms,
        right_max_deg=right_max,
        right_rms_deg=right_rms,
        max_thresh_deg=max_th,
        rms_thresh_deg=rms_th,
        check_left=check_left,
        check_right=check_right,
    )
