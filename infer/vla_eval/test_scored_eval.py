"""Tests for scored evaluation table parsing and output."""

from __future__ import annotations

from pathlib import Path

from vla_eval.evaluator import (
    TestEvaluationResult,
    _apply_exclusivity,
    _apply_scoring,
    _table_has_scores,
)
from vla_eval.excel_writer import build_output_cells
from vla_eval.table_parser import PerformanceItem, TestColumn, parse_eval_table
from vla_eval.utils import load_yaml


def test_scored_table_parses_score_column():
    table_path = Path(__file__).resolve().parents[1] / "template" / "评测表.et"
    if not table_path.is_file():
        return
    cfg = load_yaml(Path(__file__).resolve().parents[1] / "config_scored.yaml")
    table = parse_eval_table(table_path, cfg["table"])
    assert _table_has_scores(table)
    assert len(table.tests) == 12
    assert len(table.performances) == 20
    grasp = next(p for p in table.performances if p.name == "稳定准确抓取")
    assert grasp.score == 2.0
    assert grasp.stage == "抓取"


def test_exclusivity_min_score_picks_worst():
    items = [
        PerformanceItem(1, 2, "1", "定位", "10s内几乎没有动作", 0.0),
        PerformanceItem(2, 2, "", "定位", "5s内快速准确定位", 4.0),
    ]
    table = type("T", (), {"performances": items})()
    hits = {"10s内几乎没有动作": True, "5s内快速准确定位": True}
    resolved = _apply_exclusivity(
        table,
        hits,
        {"enabled": True, "group_by": "stage", "strategy": "min_score"},
    )
    assert resolved["10s内几乎没有动作"] is True
    assert resolved["5s内快速准确定位"] is False


def test_exclusivity_max_score_picks_highest():
    items = [
        PerformanceItem(1, 2, "1", "定位", "10s内几乎没有动作", 0.0),
        PerformanceItem(2, 2, "", "定位", "5s内快速准确定位", 4.0),
    ]
    table = type("T", (), {"performances": items})()
    hits = {"10s内几乎没有动作": True, "5s内快速准确定位": True}
    resolved = _apply_exclusivity(
        table,
        hits,
        {"enabled": True, "group_by": "stage", "strategy": "max_score"},
    )
    assert resolved["5s内快速准确定位"] is True
    assert resolved["10s内几乎没有动作"] is False


def test_apply_scoring_all_stages_positive():
    items = [
        PerformanceItem(1, 2, "1", "定位", "5s内快速准确定位", 4.0),
        PerformanceItem(2, 2, "2", "抓取", "稳定准确抓取", 2.0),
        PerformanceItem(3, 2, "3", "移动", "移动到盒子上方", 2.0),
        PerformanceItem(4, 2, "4", "投放", "松开", 1.0),
        PerformanceItem(5, 2, "5", "落点", "投放至盒子中心", 3.0),
        PerformanceItem(6, 2, "6", "归位", "成功归位", 2.0),
    ]
    table = type("T", (), {"performances": items})()
    test = TestColumn(col=4, header="test 1", test_id=1)
    result = TestEvaluationResult(test=test, log_dir=None)
    hits = {"5s内快速准确定位": True, "稳定准确抓取": True}
    _apply_scoring(table, result, hits, {"scoring": {"success_row": "all_stages_positive"}})
    assert result.stage_scores == {"定位": 4.0, "抓取": 2.0}
    assert result.total_score == 6.0
    assert result.success == "0"
    assert result.performance_scores["5s内快速准确定位"] == 4.0


def test_apply_scoring_success_when_all_six_stages_positive():
    stages = ["定位", "抓取", "移动", "投放", "落点", "归位"]
    items = [
        PerformanceItem(i, 2, str(i), stage, f"perf_{stage}", float(i))
        for i, stage in enumerate(stages, start=1)
    ]
    table = type("T", (), {"performances": items})()
    result = TestEvaluationResult(test=TestColumn(col=4, header="test 1", test_id=1), log_dir=None)
    hits = {item.name: True for item in items}
    _apply_scoring(table, result, hits, {"scoring": {"success_row": "all_stages_positive"}})
    assert result.success == "1"
    assert result.reward == 1.0


def test_build_output_cells_clears_template_placeholders():
    items = [
        PerformanceItem(5, 2, "1", "定位", "10s内几乎没有动作", 0.0),
        PerformanceItem(12, 2, "2", "抓取", "稳定准确抓取", 2.0),
    ]
    table = type(
        "T",
        (),
        {
            "performances": items,
            "label_rows": {"success": 25, "remark": 26},
            "source_path": Path("x.et"),
            "nrows": 30,
            "ncols": 16,
        },
    )()
    test = TestColumn(col=4, header="test 1", test_id=1)
    result = TestEvaluationResult(test=test, log_dir=None)
    result.performance_hits = {"稳定准确抓取": True}
    result.performance_scores = {"稳定准确抓取": 2.0}
    result.total_score = 2.0
    result.success = "0"
    cells = build_output_cells(table, [result], {"clear_test_cells": True, "write_remark": False})
    assert cells[(5, 4)] == ""
    assert cells[(12, 4)] == 2.0
    assert cells[(25, 4)] == "0"
    assert cells.get((26, 4), "") == ""


def test_build_output_cells_writes_scores_not_one():
    items = [
        PerformanceItem(5, 2, "1", "定位", "10s内几乎没有动作", 0.0),
    ]
    table = type(
        "T",
        (),
        {
            "performances": items,
            "label_rows": {"success": 24},
            "source_path": Path("x.et"),
            "nrows": 30,
            "ncols": 16,
        },
    )()
    test = TestColumn(col=4, header="test 1", test_id=1)
    result = TestEvaluationResult(test=test, log_dir=None)
    result.performance_hits = {"10s内几乎没有动作": True}
    result.performance_scores = {"10s内几乎没有动作": 0.0}
    result.total_score = 0.0
    result.success = "0"
    cells = build_output_cells(table, [result], {"clear_test_cells": False})
    assert cells[(5, 4)] == 0.0
    assert cells[(24, 4)] == "0"
