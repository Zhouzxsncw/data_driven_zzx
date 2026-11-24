#!/usr/bin/env python3
"""CLI entry for VLA automatic evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vla_eval.runner import run_auto_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLA 自动评估脚本")
    parser.add_argument("--table", required=True, help="真实评估表格路径 (.et/.xls/.xlsx/.csv)")
    parser.add_argument("--logs", required=True, help="推理日志根目录")
    parser.add_argument("--config", required=True, help="配置文件路径 (config.yaml)")
    parser.add_argument("--output", required=True, help="输出评估结果表格路径")
    parser.add_argument("--summary", required=True, help="统计报告输出路径 (.md 或 .xlsx)")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    output_path = Path(args.output).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    run_auto_eval(
        logs_dir=Path(args.logs).expanduser().resolve(),
        output_dir=output_path.parent,
        table_template=Path(args.table).expanduser().resolve(),
        config_path=Path(args.config).expanduser().resolve(),
        result_table_name=output_path.name,
        summary_name=summary_path.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
