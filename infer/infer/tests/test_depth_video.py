"""depth 视频转换单元测试。"""

from __future__ import annotations

import unittest

import numpy as np

from infer.recording.depth_video import depth_u16_to_video_rgb


class DepthVideoTests(unittest.TestCase):
    def test_empty_depth_returns_black_rgb(self) -> None:
        depth = np.zeros((4, 6), dtype=np.uint16)
        rgb = depth_u16_to_video_rgb(depth)
        self.assertEqual(rgb.shape, (4, 6, 3))
        self.assertTrue(np.all(rgb == 0))

    def test_valid_depth_returns_rgb(self) -> None:
        depth = np.array([[0, 500, 1000], [1500, 2000, 0]], dtype=np.uint16)
        rgb = depth_u16_to_video_rgb(depth)
        self.assertEqual(rgb.shape, (2, 3, 3))
        self.assertGreater(int(rgb[0, 1].max()), 0)


if __name__ == "__main__":
    unittest.main()
