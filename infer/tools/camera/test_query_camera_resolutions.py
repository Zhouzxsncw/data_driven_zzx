"""query_camera_resolutions 配置解析单元测试（无需真机）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.camera.query_camera_resolutions import (
    _group_by_resolution,
    resolve_camera_serial,
    StreamProfileRow,
)


class QueryCameraResolutionsTests(unittest.TestCase):
    def test_resolve_mid_from_infer_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "infer.yaml"
            path.write_text(
                """
camera_sender:
  devices:
    - serial: "left_serial"
      fps: 30
    - serial: "mid_serial"
      fps: 30
""".strip(),
                encoding="utf-8",
            )
            serial, label = resolve_camera_serial(path, camera="mid")
            self.assertEqual(serial, "mid_serial")
            self.assertIn("cam_mid", label)

    def test_resolve_mid_from_collect_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "collect.yaml"
            path.write_text(
                """
cameras:
  devices:
    - key: observation.cam_left
      serial: "left_serial"
      width: 960
      height: 540
      fps: 30
    - key: observation.cam_mid
      serial: "mid_serial"
      width: 960
      height: 540
      fps: 30
""".strip(),
                encoding="utf-8",
            )
            serial, label = resolve_camera_serial(path, camera="mid")
            self.assertEqual(serial, "mid_serial")
            self.assertEqual(label, "observation.cam_mid")

    def test_group_by_resolution(self) -> None:
        rows = [
            StreamProfileRow("color", 960, 540, 30, "rgb8"),
            StreamProfileRow("color", 960, 540, 15, "rgb8"),
            StreamProfileRow("color", 640, 480, 30, "rgb8"),
        ]
        grouped = _group_by_resolution(rows)
        labels = [g[0] for g in grouped]
        self.assertEqual(labels[0], "color 960x540")
        self.assertEqual(len(grouped[0][1]), 2)


if __name__ == "__main__":
    unittest.main()
