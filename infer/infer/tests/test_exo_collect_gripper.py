"""外骨骼数采夹爪来源：state=UDP、action=外骨骼。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from infer.exo.exo_collect import (
    GRIPPER_DIM_INDEX,
    action_15d_preserve_exo_gripper,
    observation_state_15d_with_udp_gripper,
)
from infer.exo.exo_kinematics import exo_sample_to_robot_action_15d


class _MockDirectRobot:
    def __init__(self, *, gripper_mode: str = "udp", udp_ratios: list[float] | None = None):
        self.cfg = SimpleNamespace(gripper_mode=gripper_mode)
        self._udp_ratios = udp_ratios

    def gripper_ratio_for_observation(self):
        if self._udp_ratios is None:
            return None
        return float(self._udp_ratios[0])

    def gripper_ratios_for_observation(self):
        return None if self._udp_ratios is None else list(self._udp_ratios)

    def apply_udp_gripper_to_state_15d(self, state_15d: np.ndarray) -> np.ndarray:
        out = np.asarray(state_15d, dtype=np.float64).copy()
        if self._udp_ratios is not None:
            out[GRIPPER_DIM_INDEX] = float(self._udp_ratios[0])
            if len(self._udp_ratios) > 1 and out.size > 15:
                out[15] = float(self._udp_ratios[1])
        return out


class TestExoCollectGripperSources(unittest.TestCase):
    def test_state_gripper_from_udp_not_buffer(self) -> None:
        raw = np.linspace(0.1, 0.9, 16, dtype=np.float64)
        raw[GRIPPER_DIM_INDEX] = 0.99
        raw[15] = 0.11
        robot = _MockDirectRobot(udp_ratios=[0.42, 0.55])
        state, err = observation_state_15d_with_udp_gripper(raw, robot)
        assert state is not None and err is None
        self.assertAlmostEqual(float(state[GRIPPER_DIM_INDEX]), 0.42, places=9)
        self.assertAlmostEqual(float(state[15]), 0.55, places=9)
        self.assertNotAlmostEqual(float(state[GRIPPER_DIM_INDEX]), 0.99, places=3)

    def test_state_gripper_udp_missing_drops(self) -> None:
        raw = np.zeros(16, dtype=np.float64)
        robot = _MockDirectRobot(udp_ratios=None)
        state, err = observation_state_15d_with_udp_gripper(raw, robot)
        self.assertIsNone(state)
        self.assertEqual(err, "state_gripper_udp_missing")

    def test_action_gripper_from_exo_only(self) -> None:
        joint_rad = np.zeros(17, dtype=np.float64)
        action = exo_sample_to_robot_action_15d(
            joint_rad,
            left_gripper=0.73,
            right_gripper=0.81,
        )
        self.assertAlmostEqual(float(action[GRIPPER_DIM_INDEX]), 0.73, places=9)
        self.assertAlmostEqual(float(action[15]), 0.81, places=9)
        preserved = action_15d_preserve_exo_gripper(action)
        np.testing.assert_allclose(preserved, action)

    def test_action_gripper_clamped(self) -> None:
        action = exo_sample_to_robot_action_15d(
            np.zeros(17, dtype=np.float64),
            left_gripper=2.0,
            right_gripper=-1.0,
        )
        self.assertAlmostEqual(float(action[GRIPPER_DIM_INDEX]), 1.0, places=9)
        self.assertAlmostEqual(float(action[15]), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
