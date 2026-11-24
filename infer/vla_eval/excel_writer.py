"""Write evaluation results back to Excel/CSV while preserving format."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vla_eval.evaluator import TestEvaluationResult
from vla_eval.table_parser import EvalTable, PerformanceItem
from vla_eval.utils import ensure_parent, normalize_text


def _write_xls_from_template(
    template_path: Path,
    output_path: Path,
    cell_values: Dict[Tuple[int, int], Any],
) -> None:
    import xlrd
    import xlwt
    from xlutils.copy import copy as xl_copy

    rb = xlrd.open_workbook(str(template_path), formatting_info=True)
    wb = xl_copy(rb)
    ws = wb.get_sheet(0)
    style = xlwt.XFStyle()
    alignment = xlwt.Alignment()
    alignment.wrap = 1
    style.alignment = alignment
    for (row, col), value in sorted(cell_values.items()):
        if value is None or value == "":
            continue
        ws.write(row, col, value, style)
    ensure_parent(output_path)
    wb.save(str(output_path))


def _write_csv(
    template_path: Path,
    output_path: Path,
    cell_values: Dict[Tuple[int, int], Any],
    nrows: int,
    ncols: int,
) -> None:
    rows: List[List[Any]] = []
    with template_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.reader(fp)
        rows = [list(row) for row in reader]
    while len(rows) < nrows:
        rows.append([])
    for r in range(len(rows)):
        while len(rows[r]) < ncols:
            rows[r].append("")
    for (row, col), value in cell_values.items():
        if value is None or value == "":
            continue
        rows[row][col] = value
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)


def _performance_row_map(table: EvalTable) -> Dict[str, PerformanceItem]:
    return {item.name: item for item in table.performances}


def _append_template_config_cells(
    cells: Dict[Tuple[int, int], Any],
    results: Sequence[TestEvaluationResult],
    output_cfg: Dict[str, Any],
) -> None:
    if not output_cfg.get("update_template_config") or not results:
        return
    row_map = output_cfg.get("template_config_rows") or {}
    if not row_map:
        return

    log_dir = next((item.log_dir for item in results if item.log_dir), None)
    meta: Dict[str, Any] = {}
    if log_dir is not None:
        meta_path = log_dir / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

    n_steps = meta.get("n_action_steps")
    if n_steps is not None and "n_action_steps" in row_map:
        spec = row_map["n_action_steps"]
        cells[(int(spec["row"]), int(spec["col"]))] = f"N_ACTION_STEPS_OVERRIDE = {int(n_steps)}"

    if "num_inference_steps" in row_map:
        spec = row_map["num_inference_steps"]
        cells[(int(spec["row"]), int(spec["col"]))] = "NUM_INFERENCE_STEPS_OVERRIDE = NULL"

    prompt = normalize_text(meta.get("task_prompt"))
    if prompt and "prompt" in row_map:
        spec = row_map["prompt"]
        cells[(int(spec["row"]), int(spec["col"]))] = f'提示词： "{prompt}"'

    if log_dir is not None and "paths" in row_map:
        spec = row_map["paths"]
        model_path = normalize_text(meta.get("ckpt_dir")) or "pretrained_model"
        video_path = f"{log_dir.name}/eval_videos"
        cells[(int(spec["row"]), int(spec["col"]))] = (
            f"video路径：{video_path}\n"
            f"模型路径：{model_path}"
        )


def _collect_clear_cells(
    table: EvalTable,
    results: Sequence[TestEvaluationResult],
    output_cfg: Dict[str, Any],
) -> Dict[Tuple[int, int], Any]:
    """写结果前清空 test 列里的旧值/模板占位（避免预填 0 与脏数据）。"""
    if not output_cfg.get("clear_test_cells", True):
        return {}
    clears: Dict[Tuple[int, int], Any] = {}
    test_cols = {result.test.col for result in results}
    rows_to_clear: List[int] = [item.row for item in table.performances]
    for key in ("success", "remark", "reward", "log_path", "video_path"):
        if key in table.label_rows:
            rows_to_clear.append(table.label_rows[key])
    for col in test_cols:
        for row in rows_to_clear:
            clears[(row, col)] = ""
    return clears


def build_output_cells(
    table: EvalTable,
    results: Sequence[TestEvaluationResult],
    output_cfg: Dict[str, Any],
) -> Dict[Tuple[int, int], Any]:
    cells: Dict[Tuple[int, int], Any] = dict(
        _collect_clear_cells(table, results, output_cfg)
    )
    perf_map = _performance_row_map(table)
    use_scores = any(item.score is not None for item in table.performances)
    write_remark = output_cfg.get("write_remark", not use_scores)

    for result in results:
        col = result.test.col
        for perf_name, hit in result.performance_hits.items():
            if not hit:
                continue
            item = perf_map.get(perf_name)
            if item is None:
                continue
            if use_scores and perf_name in result.performance_scores:
                cells[(item.row, col)] = result.performance_scores[perf_name]
            elif use_scores and item.score is not None:
                cells[(item.row, col)] = item.score
            else:
                cells[(item.row, col)] = 1

        if "success" in table.label_rows:
            if result.success:
                cells[(table.label_rows["success"], col)] = result.success
            elif result.total_score is not None:
                cells[(table.label_rows["success"], col)] = result.total_score
        if write_remark and "remark" in table.label_rows and result.remark:
            cells[(table.label_rows["remark"], col)] = result.remark
        if "reward" in table.label_rows and result.reward is not None:
            cells[(table.label_rows["reward"], col)] = result.reward
        if "log_path" in table.label_rows and result.log_path:
            cells[(table.label_rows["log_path"], col)] = result.log_path
        elif output_cfg.get("add_missing_columns") and result.log_path:
            pass
        if "video_path" in table.label_rows and result.video_path:
            cells[(table.label_rows["video_path"], col)] = result.video_path

    _append_template_config_cells(cells, results, output_cfg)
    return cells


def write_evaluated_table(
    table: EvalTable,
    results: Sequence[TestEvaluationResult],
    output_path: Path,
    output_cfg: Dict[str, Any],
) -> Path:
    output_path = output_path.expanduser().resolve()
    cells = build_output_cells(table, results, output_cfg)
    suffix = output_path.suffix.lower()
    source_suffix = table.source_path.suffix.lower()

    if suffix == ".csv" or source_suffix == ".csv":
        _write_csv(table.source_path, output_path, cells, table.nrows, table.ncols)
        return output_path

    if source_suffix in {".et", ".xls"} or suffix in {".et", ".xls"}:
        _write_xls_from_template(table.source_path, output_path, cells)
        return output_path

    if source_suffix in {".xlsx", ".xlsm"} or suffix in {".xlsx", ".xlsm"}:
        try:
            import openpyxl
        except ImportError as exc:
            raise RuntimeError("写入 .xlsx 需要 openpyxl") from exc
        shutil.copy2(table.source_path, output_path)
        wb = openpyxl.load_workbook(output_path)
        ws = wb.active
        for (row, col), value in cells.items():
            ws.cell(row=row + 1, column=col + 1, value=value)
        ensure_parent(output_path)
        wb.save(output_path)
        return output_path

    raise ValueError(f"不支持的表格格式: {table.source_path}")
