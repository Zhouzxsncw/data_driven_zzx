"""已有推理结果不被下一次启动清掉，并用开始时间分开。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from infer.batch.batch_infer_utils import (
    dir_has_results,
    inspect_action_step_dir,
    prepare_group_output_dir,
)
from infer.runs.index import (
    allocate_exp_dir,
    allocate_run_dir,
    latest_saved_run_dir,
    list_exp_dirs,
)


class RunDirKeepTests(unittest.TestCase):
    def test_existing_files_stay_and_new_run_is_nested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "videos" / "cam_left_wrist_task01.mp4"
            video.parent.mkdir()
            video.write_bytes(b"old-video")
            (root / "inference_summary.json").write_text("{}", encoding="utf-8")
            (root / "eval_log.jsonl").write_text("row\n", encoding="utf-8")

            removed = prepare_group_output_dir(root, full_redo=False)
            new_dir = allocate_run_dir(root)

            self.assertEqual(removed, [])
            self.assertNotEqual(new_dir.resolve(), root.resolve())
            self.assertTrue(new_dir.name.startswith("run_"))
            self.assertEqual(video.read_bytes(), b"old-video")
            self.assertTrue((root / "inference_summary.json").is_file())
            self.assertTrue((root / "eval_log.jsonl").is_file())

    def test_empty_dir_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "videos").mkdir()
            (root / "trajectory_data").mkdir()
            self.assertEqual(allocate_run_dir(root).resolve(), root.resolve())

    def test_same_second_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta.json").write_text("{}", encoding="utf-8")
            frozen = datetime(2026, 10, 21, 17, 11, 3)
            with patch("infer.runs.index.datetime") as clock:
                clock.now.return_value = frozen
                first = allocate_run_dir(root)
                (first / "run.log").write_text("first", encoding="utf-8")
                second = allocate_run_dir(root)
            (second / "run.log").write_text("second", encoding="utf-8")
            self.assertEqual(first.name, "run_20261021_171103")
            self.assertEqual(second.name, "run_20261021_171103_2")
            self.assertEqual((first / "run.log").read_text(encoding="utf-8"), "first")
            self.assertEqual(latest_saved_run_dir(root), second)

    def test_suffix_10_is_later_than_suffix_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            early = root / "run_20261021_171103_2"
            late = root / "run_20261021_171103_10"
            early.mkdir()
            late.mkdir()
            (early / "run.log").write_text("2", encoding="utf-8")
            (late / "run.log").write_text("10", encoding="utf-8")
            self.assertEqual(latest_saved_run_dir(root), late)

    def test_group_meta_alone_does_not_force_a_new_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ood_group_meta.json").write_text("{}", encoding="utf-8")
            self.assertEqual(allocate_run_dir(root).resolve(), root.resolve())
            self.assertFalse(dir_has_results(root))

    def test_newer_partial_run_is_not_hidden_by_old_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "inference_summary.json").write_text("{}", encoding="utf-8")
            (root / "videos").mkdir()
            (root / "videos" / "cam_left_wrist.mp4").write_bytes(b"old")
            newer = root / "run_20261021_180000"
            newer.mkdir()
            (newer / "run.log").write_text("partial", encoding="utf-8")
            (newer / "task_segments.jsonl").write_text(
                '{"type":"task_boundary","event":"end"}\n',
                encoding="utf-8",
            )

            self.assertFalse(dir_has_results(root))
            status = inspect_action_step_dir(root, 15, tasks_per_group=12)
            self.assertEqual(status.state, "partial")
            self.assertEqual(status.test_name, newer.name)
            self.assertEqual(status.completed_tasks, 1)
            self.assertEqual((root / "videos" / "cam_left_wrist.mp4").read_bytes(), b"old")

            fresh = allocate_run_dir(root)
            self.assertNotEqual(fresh, newer)
            self.assertTrue((root / "inference_summary.json").is_file())
            self.assertTrue((newer / "run.log").is_file())

    def test_action_step_exp_dirs_increment_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 旧扁平结果：文件直接在 action-step_10/
            (root / "inference_summary.json").write_text("{}", encoding="utf-8")
            (root / "eval_log.jsonl").write_text("old\n", encoding="utf-8")

            first = allocate_exp_dir(root)
            (first / "run.log").write_text("exp1", encoding="utf-8")
            (first / "inference_summary.json").write_text("{}", encoding="utf-8")
            second = allocate_exp_dir(root)
            (second / "run.log").write_text("exp2", encoding="utf-8")
            (second / "inference_summary.json").write_text("{}", encoding="utf-8")

            self.assertEqual(first.name, "exp1")
            self.assertEqual(second.name, "exp2")
            self.assertEqual((root / "eval_log.jsonl").read_text(encoding="utf-8"), "old\n")
            self.assertEqual((first / "run.log").read_text(encoding="utf-8"), "exp1")
            self.assertEqual([p.name for p in list_exp_dirs(root)], ["exp1", "exp2"])
            self.assertEqual(latest_saved_run_dir(root).name, "exp2")
            status = inspect_action_step_dir(root, 10, tasks_per_group=12)
            self.assertEqual(status.state, "complete")
            self.assertEqual(status.test_name, "exp2")

    def test_empty_action_step_starts_at_exp1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = allocate_exp_dir(root)
            self.assertEqual(first.name, "exp1")
            self.assertTrue(first.is_dir())


if __name__ == "__main__":
    unittest.main()
