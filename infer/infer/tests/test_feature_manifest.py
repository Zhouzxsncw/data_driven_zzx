"""feature_manifest 单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infer.config_loader import load_infer_config
from infer.recording.feature_manifest import (
    FEATURE_MANIFEST_FILENAME,
    build_and_write_feature_manifest,
    build_feature_manifest,
    format_feature_manifest_text,
)


class FeatureManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg_path = (
            Path(__file__).resolve().parents[1] / "config" / "act_left_arm_direct.yaml"
        )
        cls.cfg = load_infer_config(cfg_path)

    def test_core_highlights_gripper_source(self) -> None:
        manifest = build_feature_manifest(
            self.cfg,
            runtime={
                "n_action_steps": 25,
                "chunk_size": 50,
                "pedal_task_control": True,
                "camera_keys": [],
            },
        )
        self.assertEqual(manifest["schema_version"], 2)
        self.assertIn("core", manifest)
        self.assertIn("secondary", manifest)
        ids = {f["id"] for f in manifest["core"]}
        self.assertIn("gripper_control", ids)
        self.assertIn("gripper_state_source", ids)
        grip_src = next(f for f in manifest["core"] if f["id"] == "gripper_state_source")
        self.assertEqual(grip_src["value"], "udp")
        self.assertIn("UDP", grip_src["status"])
        self.assertIn("buffer", grip_src["effect"])
        # camera_keys=[] → 视觉关
        cam = next(f for f in manifest["core"] if f["id"] == "camera_input")
        self.assertEqual(cam["status"], "关")
        self.assertEqual(manifest["summary"]["gripper_state_source"], "udp")
        self.assertFalse(manifest["summary"]["camera_on"])

    def test_human_text_is_focused(self) -> None:
        td = Path(tempfile.mkdtemp())
        manifest = build_and_write_feature_manifest(
            self.cfg,
            td,
            runtime={"n_action_steps": 15, "pedal_task_control": True},
        )
        path = td / FEATURE_MANIFEST_FILENAME
        self.assertTrue(path.is_file())
        loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["summary"]["gripper_mode"], "udp")
        text = format_feature_manifest_text(manifest)
        self.assertIn("一眼摘要", text)
        self.assertIn("夹爪控制", text)
        self.assertIn("夹爪观测", text)
        self.assertIn("核心 — 会改变控制/观测", text)
        self.assertIn("次要 — 只影响落盘", text)
        self.assertIn("→", text)


if __name__ == "__main__":
    unittest.main()
