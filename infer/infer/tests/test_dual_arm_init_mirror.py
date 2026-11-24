"""dual_arm_init_mirror 单元测试。"""

from __future__ import annotations

import unittest

from infer.io.dual_arm_init_mirror import (
    fill_symmetric_right_in_joint_17d,
    mirror_left_arm_joints_to_right_deg,
)


class DualArmInitMirrorTests(unittest.TestCase):
    def test_zero_pose_j7_210_to_30(self) -> None:
        # robot_zero_pos: 左 j2=-90 j7=210；右 j2=90 j7=30
        left = [0.0, -90.0, 0.0, 0.0, 0.0, 0.0, 210.0]
        right = mirror_left_arm_joints_to_right_deg(left)
        self.assertEqual(right, [0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 30.0])

    def test_no_torso_dual_home(self) -> None:
        left = [-29.6, -58.6, -52.6, -47.3, -45.7, -27.4, 210.0]
        right = mirror_left_arm_joints_to_right_deg(left)
        self.assertEqual(
            [round(x, 1) for x in right],
            [29.6, 58.6, 52.6, 47.3, 45.7, 27.4, 30.0],
        )

    def test_pick_place_left_j7(self) -> None:
        left = [-103.0, -73.67, 89.0, -8.31, -22.0, -75.5, 76.4]
        right = mirror_left_arm_joints_to_right_deg(left)
        self.assertEqual(
            [round(x, 2) for x in right],
            [103.0, 73.67, -89.0, 8.31, 22.0, 75.5, 163.6],
        )

    def test_fill_17d_keeps_waist_and_left(self) -> None:
        src = [
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
        out = fill_symmetric_right_in_joint_17d(src)
        self.assertEqual(out[:10], src[:10])
        self.assertEqual(
            [round(x, 2) for x in out[10:17]],
            [103.0, 73.67, -89.0, 8.31, 22.0, 75.5, 163.6],
        )


if __name__ == "__main__":
    unittest.main()
