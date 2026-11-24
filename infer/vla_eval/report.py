"""Generate markdown / Excel summary reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from vla_eval.evaluator import TestEvaluationResult
from vla_eval.table_parser import EvalTable
from vla_eval.utils import ensure_parent, normalize_text


def _success_rate(results: Sequence[TestEvaluationResult]) -> float:
    if not results:
        return 0.0
    ok = sum(1 for item in results if item.is_success)
    return ok / len(results)


def _average_reward(results: Sequence[TestEvaluationResult]) -> float:
    values = [item.reward for item in results if item.reward is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_summary_data(table: EvalTable, results: Sequence[TestEvaluationResult]) -> Dict[str, Any]:
    performance_counts = Counter()
    failure_reasons = Counter()
    model_groups: Dict[str, List[TestEvaluationResult]] = defaultdict(list)

    for result in results:
        for name, hit in result.performance_hits.items():
            if hit:
                performance_counts[name] += 1
        if result.remark:
            failure_reasons[result.remark] += 1
        group_key = normalize_text(result.log_path) or "unknown"
        model_groups[group_key].append(result)

    return {
        "table": str(table.source_path),
        "total_tests": len(results),
        "success_count": sum(1 for item in results if item.is_success),
        "success_rate": _success_rate(results),
        "average_reward": _average_reward(results),
        "performance_counts": dict(performance_counts),
        "failure_reasons": dict(failure_reasons),
        "missing_logs": sum(1 for item in results if item.missing_reason),
        "model_groups": {
            key: {
                "count": len(items),
                "success_rate": _success_rate(items),
            }
            for key, items in model_groups.items()
        },
        "tests": [
            {
                "test_id": item.test.test_id,
                "header": item.test.header,
                "metadata": item.test.metadata,
                "success": item.success,
                "total_score": item.total_score,
                "stage_scores": dict(item.stage_scores),
                "reward": item.reward,
                "remark": item.remark,
                "log_path": item.log_path,
                "video_path": item.video_path,
                "hits": [name for name, ok in item.performance_hits.items() if ok],
                "performance_scores": dict(item.performance_scores),
            }
            for item in results
        ],
    }


def _format_suggestions(summary: Dict[str, Any]) -> List[str]:
    suggestions: List[str] = []
    if summary["missing_logs"]:
        suggestions.append(
            f"有 {summary['missing_logs']} 个 test 缺失日志，请检查 log 匹配规则或目录结构。"
        )
    top_failures = sorted(summary["failure_reasons"].items(), key=lambda x: x[1], reverse=True)
    if top_failures:
        reason, count = top_failures[0]
        suggestions.append(f"最常见失败原因: {reason}（{count} 次）")
    if summary["success_rate"] < 0.5:
        suggestions.append("成功率低于 50%，建议优先检查定位与抓取相关规则阈值。")
    low_hits = sorted(summary["performance_counts"].items(), key=lambda x: x[1])
    if low_hits:
        name, count = low_hits[0]
        suggestions.append(f"出现次数最少的表现项: {name}（{count} 次），可考虑放宽规则或补充日志字段。")
    return suggestions


def write_summary_markdown(path: Path, table: EvalTable, results: Sequence[TestEvaluationResult]) -> Path:
    summary = build_summary_data(table, results)
    suggestions = _format_suggestions(summary)
    lines = [
        "# VLA 自动评估报告",
        "",
        f"- 表格: `{summary['table']}`",
        f"- 总 test 数: {summary['total_tests']}",
        f"- 成功 test 数: {summary['success_count']}",
        f"- Success rate: {summary['success_rate']:.1%}",
        f"- 平均总分: {summary['average_reward']:.3f}",
        f"- 日志缺失 test 数: {summary['missing_logs']}",
        "",
        "## 各具体表现出现次数",
        "",
    ]
    if summary["performance_counts"]:
        for name, count in sorted(summary["performance_counts"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- （无）")

    lines.extend(["", "## 失败原因统计", ""])
    if summary["failure_reasons"]:
        for reason, count in sorted(summary["failure_reasons"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- （无）")

    lines.extend(["", "## 模型 / 日志路径对比", ""])
    for key, stats in summary["model_groups"].items():
        lines.append(f"- `{key}`: count={stats['count']}, success_rate={stats['success_rate']:.1%}")

    lines.extend(["", "## 调整建议", ""])
    for tip in suggestions:
        lines.append(f"- {tip}")

    lines.extend(["", "## Test 明细", ""])
    for item in summary["tests"]:
        stage_part = ""
        if item.get("stage_scores"):
            stage_part = " | stages=" + ", ".join(
                f"{k}:{v}" for k, v in sorted(item["stage_scores"].items())
            )
        total = item.get("total_score")
        total_part = f"total={total}" if total is not None else f"success={item['success'] or '-'}"
        lines.append(
            f"- {item['header']} | {total_part}{stage_part} | "
            f"remark={item['remark'] or '-'} | hits={', '.join(item['hits']) or '-'}"
        )

    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_summary_xlsx(path: Path, table: EvalTable, results: Sequence[TestEvaluationResult]) -> Path:
    import xlwt

    summary = build_summary_data(table, results)
    wb = xlwt.Workbook()
    ws = wb.add_sheet("summary")
    headers = [
        "test_id",
        "header",
        "metadata",
        "success",
        "reward",
        "remark",
        "log_path",
        "video_path",
        "hits",
    ]
    for col, header in enumerate(headers):
        ws.write(0, col, header)
    for row_idx, item in enumerate(summary["tests"], start=1):
        ws.write(row_idx, 0, item["test_id"])
        ws.write(row_idx, 1, item["header"])
        ws.write(row_idx, 2, item["metadata"])
        ws.write(row_idx, 3, item["success"])
        ws.write(row_idx, 4, item["reward"] if item["reward"] is not None else "")
        ws.write(row_idx, 5, item["remark"])
        ws.write(row_idx, 6, item["log_path"])
        ws.write(row_idx, 7, item["video_path"])
        ws.write(row_idx, 8, ", ".join(item["hits"]))

    ws2 = wb.add_sheet("stats")
    ws2.write(0, 0, "metric")
    ws2.write(0, 1, "value")
    stats = [
        ("total_tests", summary["total_tests"]),
        ("success_count", summary["success_count"]),
        ("success_rate", summary["success_rate"]),
        ("average_reward", summary["average_reward"]),
        ("missing_logs", summary["missing_logs"]),
    ]
    for idx, (name, value) in enumerate(stats, start=1):
        ws2.write(idx, 0, name)
        ws2.write(idx, 1, value)

    ensure_parent(path)
    wb.save(str(path))
    return path
