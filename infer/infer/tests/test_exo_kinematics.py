"""外骨骼 action 16D 与 TongGenesis 17D 互逆测试。"""

from __future__ import annotations

import unittest

import numpy as np

from infer.io.direct_robot_io import action_15d_to_tonggenesis_degrees
from infer.exo.exo_kinematics import (
    exo_joint_radians_to_degrees_17d,
    exo_sample_to_robot_action_15d,
    infer_action_15d_from_tonggenesis_degrees_17d,
)


class TestExoKinematicsRoundtrip(unittest.TestCase):
    def test_action_16d_roundtrip(self) -> None:
        action = np.linspace(-0.5, 0.5, 16, dtype=np.float64)
        action[14] = 0.65
        action[15] = 0.35
        degrees_17 = action_15d_to_tonggenesis_degrees(action)
        recovered = infer_action_15d_from_tonggenesis_degrees_17d(
            degrees_17,
            left_gripper=action[14],
            right_gripper=action[15],
        )
        np.testing.assert_allclose(recovered, action, rtol=0, atol=1e-9)

    def test_exo_rad_to_action_16d_roundtrip(self) -> None:
        joint_rad = np.linspace(-1.0, 1.0, 17, dtype=np.float64)
        left_gripper = 0.42
        right_gripper = 0.58
        degrees_17 = exo_joint_radians_to_degrees_17d(joint_rad)
        action = exo_sample_to_robot_action_15d(
            joint_rad,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
        )
        recovered_degrees = action_15d_to_tonggenesis_degrees(action)
        for i in range(3, 17):
            self.assertAlmostEqual(recovered_degrees[i], degrees_17[i], places=5)
        self.assertAlmostEqual(float(action[14]), left_gripper, places=9)
        self.assertAlmostEqual(float(action[15]), right_gripper, places=9)


if __name__ == "__main__":
    unittest.main()
