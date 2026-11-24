"""realsense_profile_utils 单元测试。"""

from __future__ import annotations

import unittest

from tools.camera.realsense_profile_utils import (
    VideoProfile,
    depth_profile_distance,
    pick_depth_profiles_for_color,
)


class RealsenseProfileUtilsTests(unittest.TestCase):
    def test_pick_closest_depth_to_960x540(self) -> None:
        profiles = [
            VideoProfile(1280, 720, 30),
            VideoProfile(848, 480, 30),
            VideoProfile(640, 480, 30),
            VideoProfile(480, 270, 30),
        ]
        ranked = pick_depth_profiles_for_color(profiles, 960, 540, 30)
        self.assertEqual(ranked[0].width, 848)
        self.assertEqual(ranked[0].height, 480)

    def test_depth_distance_prefers_matching_aspect(self) -> None:
        close = depth_profile_distance(VideoProfile(1280, 720, 30), 960, 540)
        far = depth_profile_distance(VideoProfile(640, 480, 30), 960, 540)
        self.assertLess(close, far)

    def test_prefers_fps_near_color(self) -> None:
        profiles = [
            VideoProfile(848, 480, 15),
            VideoProfile(848, 480, 30),
            VideoProfile(848, 480, 60),
        ]
        ranked = pick_depth_profiles_for_color(profiles, 960, 540, 30)
        self.assertEqual(ranked[0].fps, 30)


if __name__ == "__main__":
    unittest.main()
