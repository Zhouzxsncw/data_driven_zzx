"""gripper_state_source 配置单元测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from infer.io.direct_robot_io import DirectTongGenesisRobot
from infer.exo.exo_collect import observation_state_15d_with_udp_gripper


class _FakeUdpReceiver:
    def __init__(self, left: float, right: float | None = None) -> None:
        self._left = left
        self._right = right if right is not None else left

    def get_latest_position_for_state(self):
        return [self._left, self._right]


class TestGripperStateSource(unittest.TestCase):
    def _robot(self, *, source: str, udp_left: float = 0.42, udp_right: float = 0.55) -> DirectTongGenesisRobot:
        cfg = SimpleNamespace(
            enabled=True,
            gripper_mode="udp",
            gripper_state_source=source,
            gripper_initial_position=[0.7, 0.7],
            left_gripper_kwargs=None,
            right_gripper_kwargs=None,
        )
        robot = DirectTongGenesisRobot.__new__(DirectTongGenesisRobot)
        robot.cfg = cfg
        robot._gripper_state_receiver = _FakeUdpReceiver(udp_left, udp_right)
        robot._last_gripper_command = None
        return robot

    def test_udp_source_overwrites_buffer_gripper(self) -> None:
        robot = self._robot(source="udp", udp_left=0.42, udp_right=0.55)
        state = np.zeros(16, dtype=np.float64)
        state[14] = 0.99
        state[15] = 0.11
        out = robot.apply_observation_gripper_to_state_15d(state)
        self.assertAlmostEqual(float(out[14]), 0.42, places=9)
        self.assertAlmostEqual(float(out[15]), 0.55, places=9)

    def test_buffer_source_keeps_aligned_value(self) -> None:
        robot = self._robot(source="buffer", udp_left=0.42, udp_right=0.55)
        state = np.zeros(16, dtype=np.float64)
        state[14] = 0.99
        state[15] = 0.88
        out = robot.apply_observation_gripper_to_state_15d(state)
        self.assertAlmostEqual(float(out[14]), 0.99, places=9)
        self.assertAlmostEqual(float(out[15]), 0.88, places=9)

    def test_buffer_source_zeros_use_last_action(self) -> None:
        robot = self._robot(source="buffer")
        robot.note_gripper_command(0.63, 0.71)
        state = np.zeros(16, dtype=np.float64)
        out = robot.apply_observation_gripper_to_state_15d(state)
        self.assertAlmostEqual(float(out[14]), 0.63, places=9)
        self.assertAlmostEqual(float(out[15]), 0.71, places=9)

    def test_buffer_source_zeros_without_action_use_initial(self) -> None:
        robot = self._robot(source="buffer")
        state = np.zeros(16, dtype=np.float64)
        out = robot.apply_observation_gripper_to_state_15d(state)
        self.assertAlmostEqual(float(out[14]), 0.7, places=9)
        self.assertAlmostEqual(float(out[15]), 0.7, places=9)

    def test_buffer_source_keeps_closed_last_command(self) -> None:
        """上一拍是闭合 [0,0] 时应沿用，不能改回初始开口。"""
        robot = self._robot(source="buffer")
        robot.note_gripper_command(0.0, 0.0)
        state = np.zeros(16, dtype=np.float64)
        out = robot.apply_observation_gripper_to_state_15d(state)
        self.assertAlmostEqual(float(out[14]), 0.0, places=9)
        self.assertAlmostEqual(float(out[15]), 0.0, places=9)

    def test_udp_source_missing_packet_falls_back_last_action(self) -> None:
        robot = self._robot(source="udp")
        robot._gripper_state_receiver = SimpleNamespace(
            get_latest_position_for_state=lambda: None
        )
        robot.note_gripper_command(0.51, 0.52)
        state = np.zeros(16, dtype=np.float64)
        out = robot.apply_observation_gripper_to_state_15d(state)
        self.assertAlmostEqual(float(out[14]), 0.51, places=9)
        self.assertAlmostEqual(float(out[15]), 0.52, places=9)

    def test_exo_collect_buffer_skips_udp_requirement(self) -> None:
        robot = self._robot(source="buffer", udp_left=0.42, udp_right=0.55)
        state = np.zeros(16, dtype=np.float64)
        state[14] = 0.88
        state[15] = 0.77
        out, err = observation_state_15d_with_udp_gripper(state, robot)
        self.assertIsNone(err)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertAlmostEqual(float(out[14]), 0.88, places=9)
        self.assertAlmostEqual(float(out[15]), 0.77, places=9)

    def test_exo_collect_buffer_zeros_use_last_action(self) -> None:
        robot = self._robot(source="buffer")
        robot.note_gripper_command(0.61, 0.62)
        state = np.zeros(16, dtype=np.float64)
        out, err = observation_state_15d_with_udp_gripper(state, robot)
        self.assertIsNone(err)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertAlmostEqual(float(out[14]), 0.61, places=9)
        self.assertAlmostEqual(float(out[15]), 0.62, places=9)

    def test_exo_collect_udp_missing_drops(self) -> None:
        robot = self._robot(source="udp")
        robot._gripper_state_receiver = SimpleNamespace(
            get_latest_position_for_state=lambda: None
        )
        state = np.zeros(16, dtype=np.float64)
        out, err = observation_state_15d_with_udp_gripper(state, robot)
        self.assertIsNone(out)
        self.assertEqual(err, "state_gripper_udp_missing")


if __name__ == "__main__":
    unittest.main()
