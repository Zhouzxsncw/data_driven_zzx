"""回撤后：joint_stream 代码路径已拆除；仅保留配置解析与控制器 no-op 覆盖。"""

from __future__ import annotations

import unittest

from infer.config_loader import JointStreamConfig, _apply_controller_config, _parse_joint_stream


class JointStreamConfigTest(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        cfg = _parse_joint_stream(None)
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.trajectory_mode, 0)

    def test_parse_still_accepts_legacy_block(self) -> None:
        cfg = _parse_joint_stream(
            {
                "enabled": True,
                "hz": 100,
                "high_follow": True,
                "trajectory_mode": 0,
                "radio": 0,
                "slew_deg_per_sec": 90,
            }
        )
        self.assertIsInstance(cfg, JointStreamConfig)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.hz, 100.0)

    def test_controller_yaml_no_longer_overrides(self) -> None:
        raw = {
            "experiment_name": "some_model",
            "direct_robot": {
                "enabled": True,
                "joint_stream": {"enabled": False, "hz": 15},
            },
        }
        merged = _apply_controller_config(raw)
        self.assertIs(merged, raw)
        self.assertEqual(merged["direct_robot"]["joint_stream"]["hz"], 15)


if __name__ == "__main__":
    unittest.main()
