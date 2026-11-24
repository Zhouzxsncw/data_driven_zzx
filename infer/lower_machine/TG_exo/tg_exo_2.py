"""外骨骼关节读写封装。

默认只读（力矩关闭）；自动对齐时临时使能臂关节力矩并写目标角。
底层依赖 TongGenesis ``servo.MultiDynamixel``（下位机环境）。
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "servo.py").is_file() or (parent / "servo").is_dir():
            return parent
    return here.parents[3]


_REPO = _repo_root()
if str(_REPO) not in sys.path:
    sys.path.append(str(_REPO))

from servo import MultiDynamixel, Dynamixel  # noqa: E402

logger = logging.getLogger(__name__)

# MultiDynamixel 各版本写角接口名不一，运行时探测
_ANGLE_WRITE_CANDIDATES = (
    "goal_angle_write",
    "set_angle",
    "write_angle",
    "angle_write",
    "set_goal_angle",
    "goal_pos_write",
    "set_goal_pos",
    "write_goal_angle",
)


def _gripper_ratio(angle_deg: float, position_range) -> float:
    """将外骨骼夹爪舵机角度换算为开口比；position_range=[闭合, 张开]。"""
    min_pos = float(position_range[0])
    max_pos = float(position_range[1])
    denom = max_pos - min_pos
    if abs(denom) < 1e-9:
        return 0.0
    raw = (float(angle_deg) + 180.0) / 0.088 - min_pos
    ratio = raw / denom
    return max(0.0, min(1.0, float(ratio)))


class TGExo:
    """外骨骼：17 关节 + 左夹爪(dxl 20) + 右夹爪(dxl 21)。"""

    LEFT_GRIPPER_ANGLE_IDX = 17
    RIGHT_GRIPPER_ANGLE_IDX = 18
    ARM_JOINT_COUNT = 17

    def __init__(self, l_gripper_position, r_gripper_position, params):
        self.joint = MultiDynamixel(**params)
        n_ids = len(getattr(self.joint, "DXL_IDs", params.get("dxl_ids", [])))
        self._n_ids = max(int(n_ids), 1)
        self.l_gripper_position = l_gripper_position
        self.r_gripper_position = r_gripper_position
        self._drive_enabled = False
        self._angle_writer_name: Optional[str] = None
        self._angle_writer: Optional[Callable[..., Any]] = None
        self._release_drive()
        self.set_gripper_torque([0.3, 0.3])

    def set_gripper_torque(self, torque):
        """torque: [left] 或 [left, right]。"""
        if not torque:
            return
        ids = self.joint.DXL_IDs
        values = list(torque)
        if len(values) == 1:
            values = [values[0], values[0]]
        for idx, value in (
            (self.LEFT_GRIPPER_ANGLE_IDX, values[0]),
            (self.RIGHT_GRIPPER_ANGLE_IDX, values[1]),
        ):
            if idx >= len(ids):
                continue
            self.joint.single_torque_feedback(ids[idx], float(value))

    def read(self):
        angles = self.joint.present_angle_read()
        angles = self._limit_angle(angles)
        if len(angles) <= self.LEFT_GRIPPER_ANGLE_IDX:
            raise RuntimeError(
                f"外骨骼角度维数不足: 期望 >= {self.LEFT_GRIPPER_ANGLE_IDX + 1}，得到 {len(angles)}"
            )

        l_gripper_ratio = _gripper_ratio(
            angles[self.LEFT_GRIPPER_ANGLE_IDX], self.l_gripper_position
        )
        if len(angles) > self.RIGHT_GRIPPER_ANGLE_IDX and self.r_gripper_position is not None:
            r_gripper_ratio = _gripper_ratio(
                angles[self.RIGHT_GRIPPER_ANGLE_IDX], self.r_gripper_position
            )
        else:
            r_gripper_ratio = l_gripper_ratio

        return (
            [math.radians(angle) for angle in angles[:17]],
            l_gripper_ratio,
            r_gripper_ratio,
        )

    def read_arm_angles_deg(self) -> List[float]:
        angles = self._limit_angle(self.joint.present_angle_read())
        return [float(a) for a in angles[: self.ARM_JOINT_COUNT]]

    def enable_arm_drive(self, enable: bool) -> None:
        """使能/关闭双臂主动控制（腰 0..2 与夹爪保持被动；夹爪仍可扭矩反馈）。"""
        if enable:
            self._ensure_angle_writer()
            modes = [0] * self._n_ids
            # 只驱动左右臂（TongGenesis 17D 的 [3:17] ↔ dxl 3..16）
            for i in range(3, min(self.ARM_JOINT_COUNT, self._n_ids)):
                modes[i] = 1
            self.joint.set_torque_mode(modes)
            up = getattr(self.joint, "up_torque", None)
            if callable(up):
                up()
            self._drive_enabled = True
            print(
                f"[EXO] 双臂力矩已使能（dxl 3..16，写角接口={self._angle_writer_name}）；"
                "腰/夹爪保持被动；请勿穿戴或对抗外骨骼",
                flush=True,
            )
        else:
            self._release_drive()
            self.set_gripper_torque([0.3, 0.3])
            print("[EXO] 臂关节力矩已关闭，恢复被动读取", flush=True)

    @property
    def drive_enabled(self) -> bool:
        return self._drive_enabled

    def set_arm_angles_deg(self, target_deg_17: Sequence[float]) -> None:
        """写臂关节目标角（度）。须先 enable_arm_drive(True)。"""
        if not self._drive_enabled:
            raise RuntimeError("臂关节未使能力矩，无法写目标角")
        self._ensure_angle_writer()
        assert self._angle_writer is not None
        tgt = [float(x) for x in target_deg_17[: self.ARM_JOINT_COUNT]]
        if len(tgt) != self.ARM_JOINT_COUNT:
            raise ValueError(f"需要 {self.ARM_JOINT_COUNT} 维目标角，得到 {len(tgt)}")

        present = list(self.joint.present_angle_read())
        if len(present) < self.ARM_JOINT_COUNT:
            raise RuntimeError(f"present_angle_read 维数不足: {len(present)}")
        present[: self.ARM_JOINT_COUNT] = tgt
        self._angle_writer(present)

    def slew_arm_toward_deg(
        self,
        target_deg_17: Sequence[float],
        *,
        max_step_deg: float = 2.0,
    ) -> List[float]:
        """限速逼近目标，返回本拍实际下发角。"""
        cur = self.read_arm_angles_deg()
        tgt = [float(x) for x in target_deg_17[: self.ARM_JOINT_COUNT]]
        step = max(0.05, float(max_step_deg))
        cmd: List[float] = []
        for c, t in zip(cur, tgt):
            d = t - c
            # wrap to [-180, 180]
            d = (d + 180.0) % 360.0 - 180.0
            if d > step:
                d = step
            elif d < -step:
                d = -step
            cmd.append(c + d)
        self.set_arm_angles_deg(cmd)
        return cmd

    def end(self):
        try:
            self._release_drive()
        except Exception:
            pass
        self.joint.exit_dxl_control()

    def _release_drive(self) -> None:
        self.joint.set_torque_mode([0] * self._n_ids)
        down = getattr(self.joint, "down_torque", None)
        if callable(down):
            down()
        self._drive_enabled = False

    def _ensure_angle_writer(self) -> None:
        if self._angle_writer is not None:
            return
        for name in _ANGLE_WRITE_CANDIDATES:
            fn = getattr(self.joint, name, None)
            if callable(fn):
                self._angle_writer_name = name
                self._angle_writer = fn
                return
        public = sorted(
            m for m in dir(self.joint) if not m.startswith("_") and callable(getattr(self.joint, m, None))
        )
        raise RuntimeError(
            "MultiDynamixel 未找到写角接口 "
            f"(尝试过 {_ANGLE_WRITE_CANDIDATES})。可用方法: {public}"
        )

    def _limit_angle(self, angles):
        limited_angles = []
        for angle in angles:
            angle = (angle + 180) % 360 - 180
            limited_angles.append(angle)
        return limited_angles
