"""评估报告的人类可读导出（TXT / CSV）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

# 指标中文说明（便于真机现场阅读）
METRIC_LABELS: Dict[str, str] = {
    "runtime.sync_warning_rate": "Sync 告警占比（越低越好）",
    "runtime.mean_image_diff_sec": "State/图像时间戳平均差",
    "runtime.infer_latency_p95_sec": "推理耗时 P95",
    "runtime.tracking_error_mean": "动作跟随误差均值",
    "efficiency.episode_duration_sec": "Episode 总时长",
    "efficiency.open_loop_horizon_sec": "开环执行窗口",
    "quality.action_smoothness": "动作平滑度（越高越好）",
    "quality.gripper_closure_attempts": "夹爪闭合尝试次数",
    "safety.collision_spike_score": "碰撞/冲击安全分（越高越好）",
    "task.grasp_success": "抓取成功（需 CSV/labels）",
    "task.intention_correct": "意图正确（需 CSV/labels）",
    "task.task_success": "任务成功（需 CSV/labels）",
}

CATEGORY_LABELS: Dict[str, str] = {
    "task": "任务",
    "runtime": "运行时",
    "quality": "质量",
    "safety": "安全",
    "efficiency": "效率",
}


def sibling_report_paths(json_path: Path) -> tuple[Path, Path]:
    """与 evaluation_report.json 同目录的 .txt / .csv 路径。"""
    return json_path.with_suffix(".txt"), json_path.with_suffix(".csv")


def _metric_label(name: str) -> str:
    return METRIC_LABELS.get(name, name)


def _format_value(value: Any, unit: str = "") -> str:
    if value is None:
        return "未计算"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        if unit == "ratio":
            return f"{value * 100:.2f}%"
        if unit == "score" and 0.0 <= value <= 1.0:
            return f"{value:.3f}"
        if abs(value) >= 100 or (0 < abs(value) < 0.001 and value != 0):
            return f"{value:.4g}"
        return f"{value:.4f}"
    return str(value)


def _format_details(details: Dict[str, Any]) -> str:
    if not details:
        return ""
    parts = []
    for key, val in details.items():
        if isinstance(val, float):
            parts.append(f"{key}={val:.4g}")
        else:
            parts.append(f"{key}={val}")
    return ", ".join(parts)


def format_single_run_txt(report: Dict[str, Any]) -> str:
    """单次 test 评估报告 — 纯文本。"""
    lines: List[str] = []
    sep = "=" * 72
    test_name = report.get("test_name", "?")
    lines.append(sep)
    lines.append(f"VLA 推理评估报告 — {test_name}")
    lines.append(sep)
    lines.append(f"生成时间: {report.get('generated_at', '-')}")
    lines.append(f"实验名称: {report.get('experiment_name', '-')}")
    lines.append(f"策略类型: {report.get('policy_type', '-')}")
    lines.append(f"测试目录: {report.get('test_dir', '-')}")
    lines.append("")

    score = report.get("runtime_composite_score", report.get("vla_composite_score"))
    lines.append("【综合评分】")
    if score is not None:
        lines.append(f"  Runtime 综合分: {float(score):.3f} / 1.000  （仅自动指标，越高越好）")
    else:
        lines.append("  Runtime 综合分: 未计算（指标不足）")
    lines.append("  任务成功率等请用 run_eval_batch.py + CSV 汇总评估")
    lines.append("")

    metrics: Dict[str, Any] = report.get("metrics") or {}
    by_cat: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
    for name, md in sorted(metrics.items()):
        if not isinstance(md, dict):
            continue
        cat = md.get("category", "other")
        by_cat.setdefault(cat, []).append((name, md))

    for cat in ("task", "runtime", "efficiency", "quality", "safety", "other"):
        items = by_cat.get(cat)
        if not items:
            continue
        lines.append(f"【{CATEGORY_LABELS.get(cat, cat)}】")
        for name, md in items:
            label = _metric_label(name)
            val = _format_value(md.get("value"), md.get("unit", ""))
            unit = md.get("unit", "")
            hint = "↑" if md.get("higher_is_better") else "↓"
            detail = _format_details(md.get("details") or {})
            line = f"  {label}: {val}"
            if unit and unit not in ("ratio", "score"):
                line += f" {unit}"
            line += f"  ({hint})"
            if detail:
                line += f"  [{detail}]"
            lines.append(line)
        lines.append("")

    summary = report.get("summary") or {}
    ts = summary.get("task_success")
    if ts is not None or any(
        name.startswith("task.") for name in metrics
    ):
        lines.append("【人工标签】")
        if ts is None:
            lines.append("  任务成功 (task_success): 未标注")
        else:
            lines.append(f"  任务成功 (task_success): {'成功' if ts else '失败'}")
        lines.append("")

    diag = report.get("diagnostics") or {}
    warnings = diag.get("warnings") or []
    lines.append("【诊断】")
    if report.get("alerts_count", 0):
        lines.append(f"  告警条数: {report['alerts_count']}")
    if warnings:
        for w in warnings:
            lines.append(f"  ⚠ {w.get('code', 'WARN')}: {w.get('message', '')}")
    else:
        lines.append("  无额外告警")
    lines.append("")

    artifacts = report.get("artifacts") or {}
    if artifacts:
        lines.append("【相关文件】")
        for key, path in artifacts.items():
            lines.append(f"  {key}: {path}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


def write_single_run_csv(report: Dict[str, Any], path: Path) -> None:
    """单次 test — 每行一个指标（便于 Excel 打开）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, Any] = report.get("metrics") or {}
    fieldnames = [
        "test_name",
        "experiment_name",
        "policy_type",
        "generated_at",
        "vla_composite_score",
        "metric_name",
        "metric_label",
        "category",
        "value",
        "unit",
        "higher_is_better",
        "details",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        base = {
            "test_name": report.get("test_name", ""),
            "experiment_name": report.get("experiment_name", ""),
            "policy_type": report.get("policy_type", ""),
            "generated_at": report.get("generated_at", ""),
            "vla_composite_score": report.get("vla_composite_score"),
        }
        for name, md in sorted(metrics.items()):
            if not isinstance(md, dict):
                continue
            writer.writerow(
                {
                    **base,
                    "metric_name": name,
                    "metric_label": _metric_label(name),
                    "category": md.get("category", ""),
                    "value": md.get("value"),
                    "unit": md.get("unit", ""),
                    "higher_is_better": md.get("higher_is_better"),
                    "details": json.dumps(md.get("details") or {}, ensure_ascii=False),
                }
            )


def save_single_run_human_reports(report: Dict[str, Any], json_path: Path) -> tuple[Path, Path]:
    """写入 evaluation_report.txt 与 evaluation_report.csv。"""
    txt_path, csv_path = sibling_report_paths(json_path)
    txt_path.write_text(format_single_run_txt(report), encoding="utf-8")
    write_single_run_csv(report, csv_path)
    return txt_path, csv_path


def format_ablation_compare_txt(payload: Dict[str, Any]) -> str:
    """消融对比 — 纯文本表格。"""
    lines: List[str] = []
    sep = "=" * 72
    exp_id = payload.get("experiment_id", "?")
    lines.append(sep)
    lines.append(f"消融对比报告 — {exp_id}")
    lines.append(sep)
    lines.append(f"生成时间: {payload.get('generated_at', '-')}")
    lines.append(f"Baseline: {payload.get('baseline_run') or '-'}")
    lines.append("")

    rows: List[Dict[str, Any]] = payload.get("comparison_table") or []
    if not rows:
        lines.append("（无对比数据）")
        lines.append(sep)
        return "\n".join(lines)

    cols = [
        ("run_name", "Run"),
        ("ablation_group", "Group"),
        ("vla_composite_score", "VLA综合分"),
        ("delta.vla_composite_score", "Δ综合分"),
        ("task.task_success", "任务成功"),
        ("task.grasp_success", "抓取"),
        ("runtime.sync_warning_rate", "Sync告警率"),
        ("runtime.tracking_error_mean", "跟随误差"),
        ("efficiency.episode_duration_sec", "时长(s)"),
    ]
    header = " | ".join(label for _, label in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        cells = []
        for key, _ in cols:
            v = row.get(key)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 10 else f"{v:.2g}")
            else:
                cells.append(str(v))
        lines.append(" | ".join(cells))
    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def write_ablation_compare_csv(payload: Dict[str, Any], path: Path) -> None:
    """消融对比 — CSV（每行一个 run）。"""
    rows: List[Dict[str, Any]] = payload.get("comparison_table") or []
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    all_keys: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in all_keys})


def save_ablation_human_reports(payload: Dict[str, Any], json_path: Path) -> tuple[Path, Path]:
    txt_path = json_path.with_suffix(".txt")
    csv_path = json_path.with_suffix(".csv")
    txt_path.write_text(format_ablation_compare_txt(payload), encoding="utf-8")
    write_ablation_compare_csv(payload, csv_path)
    return txt_path, csv_path
