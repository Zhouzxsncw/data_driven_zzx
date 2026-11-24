"""batch action-step 自定义/续跑工具测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from infer.batch.batch_infer_utils import (
    action_step_dir_name,
    build_resume_plan,
    detect_batch_infer_status,
    format_action_step_execution_plan,
    is_action_step_only_batch,
    list_existing_action_step_dirs,
    resolve_action_step_dir,
    resolve_action_step_scan_targets,
    resolve_action_step_universe,
    resolve_task_prompt_presets,
    write_batch_summary,
)
from infer.config_loader import TaskPromptPreset, _parse_task_prompt_presets
from infer.core.types import PolicyType


class BatchActionStepUtilsTests(unittest.TestCase):
    def test_action_step_only_for_act(self) -> None:
        self.assertTrue(
            is_action_step_only_batch(
                configured_phase="all",
                policy_type=PolicyType.ACT,
            )
        )
        self.assertFalse(
            is_action_step_only_batch(
                configured_phase="all",
                policy_type=PolicyType.PI05,
            )
        )

    def test_scan_targets_use_disk_for_action_step_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "action-step_25").mkdir()
            (root / "action-step_30").mkdir()
            targets = resolve_action_step_scan_targets(
                root,
                [10, 20, 30, 40, 50],
                action_step_only=True,
            )
            self.assertEqual(targets, [25, 30])

    def test_write_batch_summary_merges_action_step_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "action-step_25").mkdir()
            write_batch_summary(root, {"action_step_values": [25]})
            write_batch_summary(root, {"action_step_values": [30]})
            merged = resolve_action_step_universe(root, [10, 20, 30, 40, 50])
            self.assertEqual(merged, [10, 20, 25, 30, 40, 50])

    def test_detect_status_completed_for_custom_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "action-step_25"
            out.mkdir()
            (out / "inference_summary.json").write_text("{}", encoding="utf-8")
            status = detect_batch_infer_status(
                root,
                configured_phase="action-step",
                action_step_values=[10, 20, 30, 40, 50],
                policy_type=PolicyType.ACT,
            )
            self.assertEqual(status.resume_stage, "completed")
            self.assertEqual(status.action_step_done, [25])
            self.assertEqual(status.action_step_missing, [])
            self.assertIsNone(status.interrupted_action_step)

    def test_list_existing_action_step_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "action-step_7").mkdir()
            (root / "replan-steps_15").mkdir()
            (root / "other").mkdir()
            self.assertEqual(list_existing_action_step_dirs(root), [7, 15])

    def test_dir_name_and_legacy_resolve(self) -> None:
        self.assertEqual(action_step_dir_name(5), "replan-steps_5")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "action-step_10"
            legacy.mkdir()
            self.assertEqual(resolve_action_step_dir(root, 10), legacy.resolve())
            self.assertEqual(
                resolve_action_step_dir(root, 20).name,
                "replan-steps_20",
            )

    def test_task_prompt_presets_parse_and_fallback(self) -> None:
        parsed = _parse_task_prompt_presets(
            [
                {"id": "a", "label": "A", "task_prompt": "do a"},
                {"name": "B", "task_prompt": "do b"},
            ]
        )
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].id, "a")
        self.assertEqual(parsed[1].id, "prompt_2")
        fallback = resolve_task_prompt_presets(
            [],
            default_task_prompt="yaml prompt",
        )
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0].task_prompt, "yaml prompt")
        kept = resolve_task_prompt_presets(
            [TaskPromptPreset(id="x", label="X", task_prompt="keep")],
            default_task_prompt="ignored",
        )
        self.assertEqual(kept[0].id, "x")

    def test_run_new_and_add_custom_never_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "action-step_10"
            out.mkdir()
            (out / "inference_summary.json").write_text("{}", encoding="utf-8")
            status = detect_batch_infer_status(
                root,
                configured_phase="action-step",
                action_step_values=[10, 20],
                policy_type=PolicyType.PI05,
            )
            for action in ("run_new", "add_custom"):
                plan = build_resume_plan(
                    status,
                    action=action,
                    configured_phase="action-step",
                    policy_type=PolicyType.PI05,
                )
                self.assertFalse(plan.skip_existing_action_steps, action)
                self.assertTrue(plan.run_action_step, action)

    def test_execution_plan_marks_skip_vs_new_exp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            done = root / "action-step_10"
            done.mkdir()
            (done / "exp1").mkdir()
            (done / "exp1" / "inference_summary.json").write_text("{}", encoding="utf-8")
            text_skip = format_action_step_execution_plan(
                root,
                [10],
                skip_existing=True,
                intent_label="只重做当前项",
            )
            self.assertIn("将跳过", text_skip)
            text_new = format_action_step_execution_plan(
                root,
                [10],
                skip_existing=False,
                intent_label="新测",
            )
            self.assertIn("将新测", text_new)
            self.assertIn("exp2", text_new)


if __name__ == "__main__":
    unittest.main()
