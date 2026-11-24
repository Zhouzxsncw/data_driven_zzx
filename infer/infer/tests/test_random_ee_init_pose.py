"""random_ee_init_pose 纯数学单元测试（不依赖 RealMan SDK）。"""

from __future__ import annotations

import math
import random
import unittest

from infer.io.random_ee_init_pose import (
    build_yz_parallel_target_pose,
    sample_yz_disk_offset,
)


class RandomEeInitPoseTests(unittest.TestCase):
    def test_disk_offset_within_radius(self) -> None:
        rng = random.Random(0)
        radius = 0.10
        for _ in range(200):
            dy, dz = sample_yz_disk_offset(radius, rng)
            dist = math.hypot(dy, dz)
            self.assertLessEqual(dist, radius + 1e-9)

    def test_target_pose_keeps_x_and_orientation(self) -> None:
        center = [0.4, -0.2, 0.5, 0.1, -0.2, 0.3]
        target = build_yz_parallel_target_pose(center, 0.03, -0.04)
        self.assertAlmostEqual(target[0], center[0])
        self.assertAlmostEqual(target[1], center[1] + 0.03)
        self.assertAlmostEqual(target[2], center[2] - 0.04)
        self.assertEqual(target[3:], center[3:])


if __name__ == "__main__":
    unittest.main()
