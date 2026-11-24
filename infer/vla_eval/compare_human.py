#!/usr/bin/env python3
"""Compare auto evaluation against human-labeled .et table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import xlrd

from vla_eval.evaluator import evaluate_table
from vla_eval.table_parser import parse_eval_table
from vla_eval.utils import load_yaml, normalize_text


def _human_marks(table_path: Path, table_cfg: dict) -> Dict[Tuple[str, int], bool]:
    table = parse_eval_table(table_path, table_cfg)
    import xlrd

    sh = xlrd.open_workbook(str(table_path)).sheet_by_index(0)
    marks: Dict[Tuple[str, int], bool] = {}
    for item in table.performances:
        for test in table.tests:
            if test.col >= sh.ncols:
                continue
            value = sh.cell_value(item.row, test.col)
            marks[(item.name, test.test_id)] = value in (1, 1.0, "1", "1.0")
    return marks


def _auto_marks(table_path: Path, logs_root: Path, config: dict) -> Dict[Tuple[str, int], bool]:
    table = parse_eval_table(table_path, config.get("table", {}))
    results = evaluate_table(table, logs_root, config)
    marks: Dict[Tuple[str, int], bool] = {}
    for result in results:
        for name, hit in result.performance_hits.items():
            marks[(name, result.test.test_id)] = bool(hit)
    return marks


def compare(
    human_path: Path,
    logs_root: Path,
    config_path: Path,
) -> str:
    config = load_yaml(config_path)
    table_cfg = config.get("table", {})
    human = _human_marks(human_path, table_cfg)
    auto = _auto_marks(human_path, logs_root, config)

    keys = sorted(set(human) | set(auto), key=lambda x: (x[1], x[0]))
    lines: List[str] = [
        "# 自动评估 vs 人工评估对比",
        "",
        f"- 人工表: `{human_path}`",
        f"- 日志目录: `{logs_root}`",
        f"- 配置: `{config_path}`",
        "",
        "## 逐条对比",
        "",
        "| test | 具体表现 | 人工 | 自动 | 一致 |",
        "|------|----------|------|------|------|",
    ]

    tp = fp = fn = tn = 0
    mismatches: List[str] = []
    for name, test_id in keys:
        h = human.get((name, test_id), False)
        a = auto.get((name, test_id), False)
        if h and a:
            tp += 1
        elif h and not a:
            fn += 1
            mismatches.append(f"test {test_id} | {name}: 人工=1 自动=空 (漏检)")
        elif not h and a:
            fp += 1
            mismatches.append(f"test {test_id} | {name}: 人工=空 自动=1 (误检)")
        else:
            tn += 1
        lines.append(f"| test {test_id} | {name} | {'1' if h else ''} | {'1' if a else ''} | {'✓' if h == a else '✗'} |")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0

    lines.extend(
        [
            "",
            "## 汇总",
            "",
            f"- 总单元格: {total}",
            f"- TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}",
            f"- Accuracy: {accuracy:.1%}",
            f"- Precision (标1): {precision:.1%}",
            f"- Recall (标1): {recall:.1%}",
            f"- F1: {f1:.1%}",
            "",
            "## 不一致项",
            "",
        ]
    )
    if mismatches:
        lines.extend(f"- {item}" for item in mismatches)
    else:
        lines.append("- （全部一致）")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="自动评估与人工表对比")
    parser.add_argument("--human-table", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = compare(
        Path(args.human_table),
        Path(args.logs),
        Path(args.config),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
