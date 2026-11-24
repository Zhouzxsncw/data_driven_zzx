"""数采 ROBOT_STATE 与下发/推理 J7 ±120° 对齐测试。"""

from __future__ import annotations

import math
import unittest

import numpy as np

from data_collect.robot_arm_bridge import (
    action_15d_to_tonggenesis_degrees,
    rm75_arm_degrees_to_dataset_state_rad,
)


class TestJ7StateOffset(unittest.TestCase):
    def test_j7_minus_120_before_rad(self) -> None:
        left = np.arange(7, dtype=np.float64) * 10.0
        right = np.arange(7, dtype=np.float64) * 10.0 + 1.0
        left[6] = 200.0
        right[6] = 150.0
        out = rm75_arm_degrees_to_dataset_state_rad(left, right)
        self.assertEqual(out.shape, (14,))
        self.assertAlmostEqual(out[6], math.radians(80.0), places=9)
        self.assertAlmostEqual(out[13], math.radians(30.0), places=9)
        self.assertAlmostEqual(out[0], math.radians(0.0), places=9)
        self.assertAlmostEqual(out[7], math.radians(1.0), places=9)

    def test_send_plus_120_matches_hardware_when_state_aligned(self) -> None:
        """跟踪良好时：落盘 action≈state（逻辑角），下发后机身角 = 逻辑 + 120。"""
        logical = np.zeros(16, dtype=np.float64)
        logical[6] = math.radians(40.0)
        logical[13] = math.radians(-20.0)
        cmd = action_15d_to_tonggenesis_degrees(logical)
        # 17D: 腰3 + 左7 + 右7 → 左 J7=index 9，右 J7=index 16
        self.assertAlmostEqual(cmd[9], 160.0, places=6)
        self.assertAlmostEqual(cmd[16], 100.0, places=6)
        # 机身反馈再经数采转换应回到逻辑角
        left_hw = np.array(cmd[3:10], dtype=np.float64)
        right_hw = np.array(cmd[10:17], dtype=np.float64)
        state = rm75_arm_degrees_to_dataset_state_rad(left_hw, right_hw)
        np.testing.assert_allclose(state[6], logical[6], atol=1e-9)
        np.testing.assert_allclose(state[13], logical[13], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
