"""Parse evaluation tables from Excel (.et/.xls/.xlsx) or CSV."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vla_eval.utils import (
    as_float,
    extract_test_id,
    first_non_empty,
    match_any_label,
    normalize_header,
    normalize_text,
)


@dataclass(frozen=True)
class PerformanceItem:
    row: int
    col_performance: int
    sequence: str
    stage: str
    name: str
    score: Optional[float] = None


@dataclass(frozen=True)
class TestColumn:
    col: int
    header: str
    test_id: int
    metadata: str = ""


@dataclass
class EvalTable:
    source_path: Path
    sheet_name: str
    tests: List[TestColumn]
    performances: List[PerformanceItem]
    label_rows: Dict[str, int]
    config_rows: List[int] = field(default_factory=list)
    raw_cells: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    nrows: int = 0
    ncols: int = 0

    def performance_by_name(self, name: str) -> Optional[PerformanceItem]:
        target = normalize_text(name)
        for item in self.performances:
            if normalize_text(item.name) == target:
                return item
        return None


def _read_sheet_matrix(path: Path, sheet: Any) -> Tuple[List[List[Any]], str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            rows = list(csv.reader(fp))
        return rows, path.stem

    if suffix in {".xlsx", ".xlsm"}:
        try:
            import openpyxl
        except ImportError as exc:
            raise RuntimeError(
                "读取 .xlsx 需要 openpyxl，请执行: pip install openpyxl"
            ) from exc
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        if isinstance(sheet, int):
            ws = wb.worksheets[sheet]
        else:
            ws = wb[sheet]
        rows = [[cell.value for cell in row] for row in ws.iter_rows(values_only=True)]
        wb.close()
        return rows, ws.title

    import xlrd

    wb = xlrd.open_workbook(str(path))
    if isinstance(sheet, int):
        sh = wb.sheet_by_index(sheet)
    else:
        sh = wb.sheet_by_name(sheet)
    rows = [
        [sh.cell_value(r, c) for c in range(sh.ncols)]
        for r in range(sh.nrows)
    ]
    return rows, sh.name


def _resolve_sheet_name(table_cfg: Dict[str, Any], available: Sequence[str]) -> Any:
    if "sheet_name" in table_cfg:
        return table_cfg["sheet_name"]
    if "sheet_index" in table_cfg:
        return int(table_cfg["sheet_index"])
    preferred = table_cfg.get("preferred_sheet_names") or []
    for name in preferred:
        if name in available:
            return name
    return 0


def _looks_like_config_row(performance_name: str, patterns: Sequence[str]) -> bool:
    text = normalize_text(performance_name)
    if not text:
        return True
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def parse_eval_table(path: Path, table_cfg: Dict[str, Any]) -> EvalTable:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"评估表格不存在: {path}")

    if path.suffix.lower() == ".csv":
        rows, sheet_name = _read_sheet_matrix(path, 0)
    else:
        import xlrd

        wb = xlrd.open_workbook(str(path))
        sheet_selector = _resolve_sheet_name(table_cfg, wb.sheet_names())
        rows, sheet_name = _read_sheet_matrix(path, sheet_selector)

    nrows = len(rows)
    ncols = max((len(row) for row in rows), default=0)

    test_pattern = table_cfg.get("test_header_pattern", r"test\s*(\d+)")
    label_rows_cfg: Dict[str, Sequence[str]] = table_cfg.get(
        "label_rows",
        {
            "success": ["是否成功"],
            "remark": ["备注"],
            "reward": ["reward", "score", "得分"],
            "log_path": ["log_path", "日志路径"],
            "video_path": ["video_path", "video路径", "对应video路径"],
        },
    )
    column_labels = table_cfg.get(
        "column_labels",
        {
            "sequence": ["序号"],
            "stage": ["机械臂状态"],
            "performance": ["具体表现"],
            "score": ["分值", "分数", "score"],
        },
    )
    exclude_patterns = table_cfg.get(
        "exclude_performance_patterns",
        [
            r"N_ACTION_STEPS",
            r"NUM_INFERENCE",
            r"video路径",
            r"模型路径",
            r"提示词",
            r"TEXT PROMPT",
        ],
    )
    metadata_row = table_cfg.get("metadata_row")
    label_search_cols = int(table_cfg.get("label_search_cols", 4))

    raw_cells: Dict[Tuple[int, int], Any] = {}
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if normalize_text(value):
                raw_cells[(r, c)] = value

    # Locate label columns by header row.
    header_row_idx: Optional[int] = None
    col_sequence: Optional[int] = None
    col_stage: Optional[int] = None
    col_performance: Optional[int] = None
    col_score: Optional[int] = None
    for r, row in enumerate(rows):
        hits = 0
        local_seq = local_stage = local_perf = local_score = None
        for c, value in enumerate(row):
            text = normalize_text(value)
            if match_any_label(text, column_labels.get("sequence", [])):
                local_seq = c
                hits += 1
            if match_any_label(text, column_labels.get("stage", [])):
                local_stage = c
                hits += 1
            if match_any_label(text, column_labels.get("performance", [])):
                local_perf = c
                hits += 1
            if match_any_label(text, column_labels.get("score", [])):
                local_score = c
                hits += 1
        if hits >= 2 and local_perf is not None:
            header_row_idx = r
            col_sequence, col_stage, col_performance = local_seq, local_stage, local_perf
            col_score = local_score
            break

    if header_row_idx is None or col_performance is None:
        raise ValueError("无法识别表头行，请在 config.yaml 的 table 段检查 column_labels 配置")

    tests: List[TestColumn] = []
    header_row = rows[header_row_idx]
    for c, value in enumerate(header_row):
        header = normalize_text(value)
        if not header:
            continue
        test_id = extract_test_id(header, test_pattern)
        if test_id is None:
            continue
        metadata = ""
        if metadata_row is not None and metadata_row < nrows:
            metadata = normalize_text(rows[metadata_row][c] if c < len(rows[metadata_row]) else "")
        tests.append(TestColumn(col=c, header=header, test_id=test_id, metadata=metadata))

    if not tests:
        raise ValueError("未找到任何 test 列，请检查 test_header_pattern 或表头格式")

    max_test_id = table_cfg.get("max_test_id")
    if max_test_id is not None:
        tests = [test for test in tests if test.test_id <= int(max_test_id)]

    label_rows: Dict[str, int] = {}
    for r, row in enumerate(rows):
        label_text = first_non_empty(row[:label_search_cols])
        if not label_text:
            continue
        for key, labels in label_rows_cfg.items():
            if match_any_label(label_text, labels):
                label_rows[key] = r

    performances: List[PerformanceItem] = []
    config_rows: List[int] = []
    current_sequence = ""
    current_stage = ""
    for r in range(header_row_idx + 1, nrows):
        if r in label_rows.values():
            continue
        row = rows[r] if r < len(rows) else []
        perf_name = normalize_text(row[col_performance] if col_performance < len(row) else "")
        if not perf_name:
            continue
        if _looks_like_config_row(perf_name, exclude_patterns):
            config_rows.append(r)
            continue
        seq_val = normalize_text(row[col_sequence] if col_sequence is not None and col_sequence < len(row) else "")
        stage_val = normalize_text(row[col_stage] if col_stage is not None and col_stage < len(row) else "")
        if seq_val:
            current_sequence = seq_val
        if stage_val:
            current_stage = stage_val
        score_val: Optional[float] = None
        if col_score is not None and col_score < len(row):
            score_val = as_float(row[col_score])
        performances.append(
            PerformanceItem(
                row=r,
                col_performance=col_performance,
                sequence=current_sequence,
                stage=current_stage,
                name=perf_name,
                score=score_val,
            )
        )

    return EvalTable(
        source_path=path,
        sheet_name=sheet_name,
        tests=sorted(tests, key=lambda item: item.test_id),
        performances=performances,
        label_rows=label_rows,
        config_rows=config_rows,
        raw_cells=raw_cells,
        nrows=nrows,
        ncols=ncols,
    )
