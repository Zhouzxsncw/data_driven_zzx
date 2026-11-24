"""WPS action-step 评分表模板与预填逻辑回归测试。"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from infer.batch.feedback_score_sheet import (
    ACTION_STEP_TEMPLATE,
    FeedbackScoreSheetSpec,
    _ACTION_STEP_FIELD_NEEDLES,
    _ACTION_STEP_FORMULA_CELL_COUNT,
    _cell_text_ws,
    _load_workbook,
    _scan_action_step_template_cells,
    count_workbook_formula_cells,
    prepare_feedback_score_sheet,
)


class TestFeedbackScoreSheetTemplate(unittest.TestCase):
    def test_action_step_template_placeholders_exist(self) -> None:
        self.assertTrue(
            ACTION_STEP_TEMPLATE.is_file(),
            f"缺少模板: {ACTION_STEP_TEMPLATE}",
        )
        wb = _load_workbook(ACTION_STEP_TEMPLATE)
        ws = wb.active
        cells = _scan_action_step_template_cells(ws)
        for field in _ACTION_STEP_FIELD_NEEDLES:
            self.assertIn(field, cells, f"模板缺少占位字段: {field}")
            row, col = cells[field]
            text = _cell_text_ws(ws, row, col)
            self.assertTrue(text, f"{field} 单元格为空 ({row},{col})")
        wb.close()

    def test_action_step_template_has_score_formulas(self) -> None:
        self.assertGreaterEqual(
            count_workbook_formula_cells(ACTION_STEP_TEMPLATE),
            _ACTION_STEP_FORMULA_CELL_COUNT,
            "模板缺少 总分/任务进度 公式",
        )

    def test_prepare_action_step_sheet_prefills_config(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="score_sheet_test_"))
        try:
            spec = FeedbackScoreSheetSpec(
                test_dir=tmp / "action-step_25",
                n_action_steps=25,
                task_prompt="Use the left gripper to pick up the toy and put into the yellow box.",
                model_path=Path("/tmp/pretrained_model"),
                num_inference_steps=10,
                phase="action-step",
            )
            dest = prepare_feedback_score_sheet(spec)
            self.assertTrue(dest.is_file())
            self.assertEqual(dest.suffix.lower(), ".xlsx")

            wb = _load_workbook(dest)
            ws = wb.active
            cells = _scan_action_step_template_cells(ws)
            self.assertIn(
                "N_ACTION_STEPS_OVERRIDE = 25",
                _cell_text_ws(ws, *cells["n_action_steps"]),
            )
            self.assertIn(
                "NUM_INFERENCE_STEPS_OVERRIDE = 10",
                _cell_text_ws(ws, *cells["num_inference_steps"]),
            )
            prompt = _cell_text_ws(ws, *cells["prompt"])
            self.assertIn("pick up the toy", prompt)
            paths = _cell_text_ws(ws, *cells["paths"])
            self.assertIn("video路径", paths)
            self.assertIn("模型路径", paths)
            self.assertIn("/tmp/pretrained_model", paths)
            wb.close()

            self.assertGreaterEqual(
                count_workbook_formula_cells(dest),
                _ACTION_STEP_FORMULA_CELL_COUNT,
                "预填后应保留模板中的 总分/任务进度 公式",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
