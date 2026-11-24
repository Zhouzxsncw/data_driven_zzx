"""WALL-OSS 菜单上限必须来自 ckpt chunk，不能写死。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from infer.core.types import PolicyType
from tools.policy.policy_batch_support import (
    _horizon_from_mapping,
    _horizon_from_yaml_text,
    peek_websocket_action_horizon,
    read_action_horizon,
)


class WallossChunkHorizonTest(unittest.TestCase):
    def test_horizon_from_nested_task_action_horizon(self) -> None:
        self.assertEqual(
            _horizon_from_mapping(
                {"_raw_yaml": {"task": {"action_horizon": 24, "action_horizon_flow": 24}}}
            ),
            24,
        )
        self.assertEqual(_horizon_from_mapping({"task": {"action_horizon": 16}}), 16)
        self.assertEqual(_horizon_from_mapping({"chunk_size": 50}), 50)
        self.assertEqual(
            _horizon_from_mapping(
                {"name": "pi05_aloha", "model": {"action_horizon": 100, "action_dim": 32}}
            ),
            100,
        )

    def test_read_action_horizon_from_walloss_config_yml(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td)
            (ckpt / "config.yml").write_text(
                "task:\n  action_horizon: 24\n  action_horizon_flow: 24\n",
                encoding="utf-8",
            )
            self.assertEqual(read_action_horizon(ckpt, PolicyType.WALLOSS), 24)

    def test_yaml_python_tuple_tag_still_reads_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td)
            (ckpt / "config.yml").write_text(
                "hyperparams:\n  optimizer:\n    betas: !!python/tuple\n    - 0.9\n    - 0.95\n"
                "task:\n  action_horizon: 24\n",
                encoding="utf-8",
            )
            self.assertEqual(read_action_horizon(ckpt, PolicyType.WALLOSS), 24)

    def test_yaml_text_fallback(self) -> None:
        text = "task:\n  action_horizon: 18\n"
        self.assertEqual(_horizon_from_yaml_text(text), 18)

    def test_read_action_horizon_from_openpi_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td)
            (ckpt / "config.json").write_text(
                '{"name": "pi05_aloha", "model": {"action_horizon": 100}}',
                encoding="utf-8",
            )
            self.assertEqual(read_action_horizon(ckpt, PolicyType.PI05), 100)

    def test_peek_websocket_horizon_offline(self) -> None:
        self.assertIsNone(peek_websocket_action_horizon("127.0.0.1", 9, timeout_sec=0.2))
