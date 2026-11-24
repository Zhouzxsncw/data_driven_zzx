"""checkpoint 训练 loss 终端行。"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from infer.logging_setup import _TTYLossFormatter
from infer.recording.ckpt_loss_notice import (
    notice_for_ckpt,
    parse_client_yaml,
)
from infer.recording.feature_manifest import format_feature_manifest_text


def _write_trainer_state(directory: Path, loss: float, step: int, extra: dict | None = None) -> None:
    row = {"loss": loss, "step": step}
    if extra:
        row.update(extra)
    payload = {"log_history": [row], "global_step": step}
    (directory / "trainer_state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class CkptLossNoticeTest(unittest.TestCase):
    def test_gr00t_high_scalar_is_red_and_plain_has_no_ansi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint-10000"
            ckpt.mkdir()
            _write_trainer_state(ckpt, 0.9178, 10000)
            notice = notice_for_ckpt(ckpt, "gr00t")
        self.assertEqual(notice.verdict, "推荐不拉")
        self.assertEqual(notice.color, "red")
        self.assertAlmostEqual(notice.value or 0, 0.9178)
        self.assertEqual(notice.step, 10000)
        self.assertIn("LOSS ckpt=checkpoint-10000", notice.plain)
        self.assertNotIn("\033", notice.plain)
        self.assertIn("\033[31", notice.ansi)
        self.assertIn("推荐不拉", notice.ansi)

    def test_pi05_bands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "00036_pi05" / "006300" / "pretrained_model"
            root.mkdir(parents=True)
            history = {
                "steps": [100, 6300, 20000],
                "losses": [0.2, 0.0062, 0.04],
            }
            (root.parents[1] / "train_loss_history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )
            ok = notice_for_ckpt(root, "pi05")
            self.assertEqual(ok.verdict, "可以拉")
            self.assertEqual(ok.color, "green")
            self.assertEqual(ok.step, 6300)

            history["losses"][1] = 0.015
            (root.parents[1] / "train_loss_history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )
            mid = notice_for_ckpt(root, "pi05")
            self.assertEqual(mid.verdict, "偏高")
            self.assertEqual(mid.color, "yellow")

            history["losses"][1] = 0.043
            (root.parents[1] / "train_loss_history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )
            bad = notice_for_ckpt(root, "pi05")
            self.assertEqual(bad.verdict, "推荐不拉")
            self.assertEqual(bad.color, "red")

    def test_history_far_from_ckpt_step_is_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp" / "checkpoint-100000"
            root.mkdir(parents=True)
            (root.parent / "train_loss_history.json").write_text(
                json.dumps({"steps": [10560], "losses": [0.0942]}),
                encoding="utf-8",
            )
            notice = notice_for_ckpt(root, "gr00t")
        self.assertEqual(notice.verdict, "未找到")
        self.assertNotIn("0.0942", notice.plain)

    def test_wam_uses_action_not_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint-3000"
            ckpt.mkdir()
            _write_trainer_state(
                ckpt,
                0.944,
                3000,
                {"flow_matching_loss_action": 0.00755, "video_loss": 0.08},
            )
            ok = notice_for_ckpt(ckpt, "cosmos3")
            self.assertEqual(ok.metric_label, "action")
            self.assertAlmostEqual(ok.value or 0, 0.00755)
            self.assertEqual(ok.verdict, "可以拉")

            _write_trainer_state(ckpt, 0.95, 3000)
            missing = notice_for_ckpt(ckpt, "cosmos3")
            self.assertEqual(missing.verdict, "未拆action")
            self.assertEqual(missing.color, "yellow")
            self.assertIn("0.95", missing.plain)

            _write_trainer_state(
                ckpt, 0.3, 3000, {"flow_matching_loss_action": 0.03}
            )
            bad = notice_for_ckpt(ckpt, "cosmos3")
            self.assertEqual(bad.verdict, "推荐不拉")
            self.assertAlmostEqual(bad.value or 0, 0.03)

    def test_tau_total_only_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "step120000"
            ckpt.mkdir()
            _write_trainer_state(ckpt, 0.32, 120000)
            bad = notice_for_ckpt(ckpt, "tau0wm")
            self.assertEqual(bad.verdict, "推荐不拉")
            _write_trainer_state(ckpt, 0.137, 120000)
            ok = notice_for_ckpt(ckpt, "tau0wm")
            self.assertEqual(ok.verdict, "可以拉")

    def test_tail_parse_ignores_earlier_action_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint-10000"
            ckpt.mkdir()
            blob = (
                '{"training/action_loss": 9.9, "step": 10}\n'
                + (" " * 2000)
                + '{"loss": 0.9178, "step": 10000}\n'
            )
            (ckpt / "trainer_state.json").write_text(blob, encoding="utf-8")
            notice = notice_for_ckpt(ckpt, "gr00t")
        self.assertAlmostEqual(notice.value or 0, 0.9178)
        self.assertEqual(notice.verdict, "推荐不拉")
        self.assertNotIn("9.9", notice.plain)

    def test_manifest_line_is_plain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint-10000"
            ckpt.mkdir()
            _write_trainer_state(ckpt, 0.9178, 10000)
            text = format_feature_manifest_text(
                {
                    "inputs": {
                        "experiment_name": "t",
                        "policy": {
                            "type": "gr00t",
                            "ckpt_dir": str(ckpt),
                            "task_prompt": "x",
                        },
                        "session": {},
                        "robot": {},
                    },
                    "summary": {},
                }
            )
        self.assertIn("一眼摘要", text)
        self.assertIn("推荐不拉", text)
        self.assertNotIn("\033", text)

    def test_tty_formatter_colors_only_loss_line(self) -> None:
        class _TTY:
            def isatty(self) -> bool:
                return True

        formatter = logging.Formatter("%(message)s")
        colored = _TTYLossFormatter(formatter, _TTY())
        bad = logging.LogRecord("n", logging.INFO, "", 0, "LOSS x 推荐不拉", (), None)
        other = logging.LogRecord("n", logging.INFO, "", 0, "ckpt: /tmp", (), None)
        self.assertIn("\033[31", colored.format(bad))
        self.assertNotIn("\033", colored.format(other))

        class _Pipe:
            def isatty(self) -> bool:
                return False

        plain = _TTYLossFormatter(formatter, _Pipe())
        self.assertNotIn("\033", plain.format(bad))

    def test_parse_client_yaml_type_nearest_ckpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "client.yaml"
            path.write_text(
                "engine:\n  type: native\npolicy:\n  type: gr00t\n"
                "  ckpt_dir: /tmp/checkpoint-10000\n",
                encoding="utf-8",
            )
            ckpt, policy = parse_client_yaml(path)
        self.assertEqual(policy, "gr00t")
        self.assertEqual(ckpt, "/tmp/checkpoint-10000")


if __name__ == "__main__":
    unittest.main()
