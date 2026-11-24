"""camera_tcp_sender 纯逻辑单元测试（无需真机）。"""

from __future__ import annotations

import unittest

import numpy as np

from infer.config_loader import CameraSenderConfig
from infer.io.camera_tcp_sender import _fps_candidates, _resize_bgr, _resize_depth
from tools.camera.checkpoint_camera_resolution import needs_output_resize


class _DeviceStub:
    def __init__(
        self,
        *,
        width: int | None,
        height: int | None,
        model_width: int | None = None,
        model_height: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.model_width = model_width
        self.model_height = model_height


class CameraTcpSenderTests(unittest.TestCase):
    def test_fps_candidates_prefers_requested(self) -> None:
        self.assertEqual(_fps_candidates(30), [30, 15, 60, 6])

    def test_fps_candidates_dedup(self) -> None:
        self.assertEqual(_fps_candidates(15), [15, 30, 60, 6])

    def test_enable_depth_defaults_true(self) -> None:
        self.assertTrue(CameraSenderConfig().enable_depth)

    def test_resize_bgr_to_model_shape(self) -> None:
        src = np.zeros((480, 640, 3), dtype=np.uint8)
        out = _resize_bgr(src, 224, 224)
        self.assertEqual(out.shape, (224, 224, 3))

    def test_resize_depth_to_model_shape(self) -> None:
        src = np.zeros((480, 640), dtype=np.uint16)
        out = _resize_depth(src, 224, 224)
        self.assertEqual(out.shape, (224, 224))

    def test_needs_output_resize_when_capture_differs_from_model(self) -> None:
        dev = _DeviceStub(width=640, height=480, model_width=224, model_height=224)
        self.assertTrue(needs_output_resize(dev))


if __name__ == "__main__":
    unittest.main()
