"""Programmatic entry for VLA automatic evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vla_eval.evaluator import evaluate_table
from vla_eval.excel_writer import write_evaluated_table
from vla_eval.report import write_summary_markdown, write_summary_xlsx
from vla_eval.table_parser import parse_eval_table
from vla_eval.utils import ensure_parent, load_yaml

logger = logging.getLogger(__name__)

_DEFAULT_TABLE = Path(__file__).resolve().parent / "template" / "评测表.et"
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_scored.yaml"


@dataclass(frozen=True)
class VlaAutoEvalResult:
    table_path: Path
    summary_path: Path
    output_dir: Path
    test_count: int
    success_count: int


def resolve_default_table_template(path: Optional[Path] = None) -> Path:
    candidate = (path or _DEFAULT_TABLE).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"VLA 评估表模板不存在: {candidate}")
    return candidate


def resolve_default_eval_config(path: Optional[Path] = None) -> Path:
    candidate = (path or _DEFAULT_CONFIG).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"VLA 评估配置不存在: {candidate}")
    return candidate


def run_auto_eval(
    *,
    logs_dir: Path,
    output_dir: Path,
    table_template: Optional[Path] = None,
    config_path: Optional[Path] = None,
    result_table_name: str = "eval_result.et",
    summary_name: str = "summary.md",
) -> VlaAutoEvalResult:
    """对单个 action-step 目录运行自动评估，输出 eval_result.et 与 summary.md。"""
    logs_dir = logs_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"推理日志目录不存在: {logs_dir}")

    table_path = resolve_default_table_template(table_template)
    eval_config_path = resolve_default_eval_config(config_path)
    config = load_yaml(eval_config_path)
    table_cfg = config.get("table", {})
    output_cfg = dict(config.get("output", {}))

    logger.info("VLA 自动评估: 表格模板 %s", table_path)
    logger.info("VLA 自动评估: 日志目录 %s", logs_dir)
    table = parse_eval_table(table_path, table_cfg)
    results = evaluate_table(table, logs_dir, config)

    ensure_parent(output_dir / result_table_name)
    result_table = output_dir / result_table_name
    summary_path = output_dir / summary_name

    write_evaluated_table(table, results, result_table, output_cfg)
    if summary_path.suffix.lower() == ".xlsx":
        write_summary_xlsx(summary_path, table, results)
    else:
        write_summary_markdown(summary_path, table, results)

    success_count = sum(1 for item in results if item.is_success)
    logger.info(
        "VLA 自动评估完成: %d/%d tests 成功, 输出 %s",
        success_count,
        len(results),
        output_dir,
    )
    return VlaAutoEvalResult(
        table_path=result_table,
        summary_path=summary_path,
        output_dir=output_dir,
        test_count=len(results),
        success_count=success_count,
    )
