"""数采夹爪来源拆分测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from data_collect.frame_gripper_utils import prepare_robot_packets_for_collect
from data_collect.modalities import ModalityGroupConfig, ModalityKind, ModalityToggle


def _left_arm_gripper_group() -> ModalityGroupConfig:
    on = ModalityToggle(enabled=True, required=True, fallback="drop_frame")
    off = ModalityToggle(enabled=False, required=False, fallback="drop_frame")
    return ModalityGroupConfig(
        left_arm=on,
        right_arm=off,
        left_gripper=on,
        right_gripper=off,
        right_hand=off,
    )


class _MockGripperReceiver:
    def __init__(self, left: float | None):
        self._left = left

    def left_ratio_if_fresh(self):
        return self._left


class _MockExoReceiver:
    def __init__(self, left: float):
        self._left = left

    def latest(self):
        return SimpleNamespace(left_gripper=self._left)


class TestFrameGripperUtils(unittest.TestCase):
    def test_state_udp_action_exo(self) -> None:
        # 16D 协议：14 关节 + 左/右夹爪（索引 14/15，见 common/robot_packet_gripper.py）。
        state = np.zeros(16, dtype=np.float64)
        state[14] = 0.99
        action = np.zeros(16, dtype=np.float64)
        action[14] = 0.11
        out_s, out_a, err = prepare_robot_packets_for_collect(
            state,
            action,
            state_modalities=_left_arm_gripper_group(),
            action_modalities=_left_arm_gripper_group(),
            gripper_receiver=_MockGripperReceiver(0.42),
            exo_receiver=_MockExoReceiver(0.73),
        )
        assert err is None and out_s is not None and out_a is not None
        self.assertAlmostEqual(float(out_s[14]), 0.42, places=9)
        self.assertAlmostEqual(float(out_a[14]), 0.73, places=9)

    def test_missing_udp_drops(self) -> None:
        state = np.zeros(16, dtype=np.float64)
        action = np.zeros(16, dtype=np.float64)
        out_s, out_a, err = prepare_robot_packets_for_collect(
            state,
            action,
            state_modalities=_left_arm_gripper_group(),
            action_modalities=_left_arm_gripper_group(),
            gripper_receiver=_MockGripperReceiver(None),
            exo_receiver=_MockExoReceiver(0.5),
        )
        self.assertIsNone(out_s)
        self.assertEqual(err, "state_gripper_udp_missing")


if __name__ == "__main__":
    unittest.main()
