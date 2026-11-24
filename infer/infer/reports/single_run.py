"""单次 run / 消融实验报告生成。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from infer.core.types import MetricValue, RunArtifacts
from infer.metrics.registry import compute_metrics, save_metrics_report
from infer.reports.human_readable import save_single_run_human_reports


def build_single_run_report(
    test_dir: Path,
    *,
    metric_names: Optional[List[str]] = None,
    experiment_name: str = "",
    policy_type: str = "",
    save: bool = True,
    report_filename: str = "evaluation_report.json",
    sync_warn_rate_threshold: float = 0.30,
    labels_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """对单个 test 目录生成评估报告。"""
    from infer.core.types import MetricContext

    test_dir = test_dir.resolve()
    ctx = MetricContext.from_test_dir(test_dir)
    if labels_file is not None:
        ctx.labels_path = Path(labels_file).expanduser()
    ctx.load()
    metrics = compute_metrics(ctx, metric_names)
    alerts_path = test_dir / "alerts.jsonl"
    alert_count = 0
    if alerts_path.is_file():
        alert_count = sum(1 for _ in alerts_path.open("r", encoding="utf-8"))

    report: Dict[str, Any] = {
        "report_type": "single_run",
        "generated_at": datetime.now().astimezone().isoformat(),
        "experiment_name": experiment_name,
        "policy_type": policy_type,
        "test_dir": str(test_dir),
        "test_name": test_dir.name,
        "metrics": {k: v.to_dict() for k, v in metrics.items()},
        "alerts_count": alert_count,
        "artifacts": {
            "machine_flow": str(test_dir / "machine_flow.jsonl"),
            "meta": str(test_dir / "meta.json"),
            "videos": str(test_dir / "videos"),
            "trajectory": str(test_dir / "trajectory_data"),
        },
    }

    # 单次 test：仅基于可自动计算的 runtime 指标综合分
    report["runtime_composite_score"] = _runtime_composite(metrics)
    report["vla_composite_score"] = report["runtime_composite_score"]

    diagnostics: Dict[str, Any] = {"warnings": [], "notes": []}
    sync_metric = metrics.get("runtime.sync_warning_rate")
    if sync_metric is not None:
        rate = float(sync_metric.value)
        if rate > sync_warn_rate_threshold:
            diagnostics["warnings"].append(
                {
                    "code": "HIGH_SYNC_WARN_RATE",
                    "message": (
                        f"sync 告警事件占比 {rate:.1%} 超过阈值 {sync_warn_rate_threshold:.0%}，"
                        "建议检查时钟标定或网络延迟"
                    ),
                    "value": rate,
                    "threshold": sync_warn_rate_threshold,
                    "detail": sync_metric.details,
                }
            )
    report["diagnostics"] = diagnostics

    if save:
        out = test_dir / report_filename
        save_metrics_report(metrics, out)
        # 附加元信息
        existing = json.loads(out.read_text(encoding="utf-8"))
        existing.update({k: v for k, v in report.items() if k != "metrics"})
        out.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        txt_path, csv_path = save_single_run_human_reports(existing, out)
        report["human_report_txt"] = str(txt_path)
        report["human_report_csv"] = str(csv_path)

    return report


def _runtime_composite(metrics: Dict[str, MetricValue]) -> Optional[float]:
    """基于 machine_flow 可自动指标的综合分（0~1），不含任务标签。"""
    weights = {
        "runtime.sync_warning_rate": (-1.0, 0.20),
        "runtime.tracking_error_mean": (-1.0, 0.20),
        "quality.action_smoothness": (1.0, 0.25),
        "safety.collision_spike_score": (1.0, 0.20),
        "efficiency.open_loop_horizon_sec": (-1.0, 0.15),
    }
    return _weighted_composite(metrics, weights)


def _vla_composite(metrics: Dict[str, MetricValue]) -> Optional[float]:
    """含任务标签时的 VLA 综合分（用于批量评估）。"""
    weights = {
        "runtime.sync_warning_rate": (-1.0, 0.12),
        "runtime.tracking_error_mean": (-1.0, 0.12),
        "quality.action_smoothness": (1.0, 0.16),
        "safety.collision_spike_score": (1.0, 0.12),
        "efficiency.open_loop_horizon_sec": (-1.0, 0.08),
        "task.task_success": (1.0, 0.40),
    }
    return _weighted_composite(metrics, weights)


def _weighted_composite(
    metrics: Dict[str, MetricValue],
    weights: Dict[str, tuple[float, float]],
) -> Optional[float]:
    score = 0.0
    weight_sum = 0.0
    for key, (direction, w) in weights.items():
        if key not in metrics:
            continue
        v = metrics[key].value
        if key == "runtime.sync_warning_rate":
            norm = max(0.0, 1.0 - min(v, 1.0))
        elif key == "runtime.tracking_error_mean":
            norm = max(0.0, 1.0 - min(v / 0.5, 1.0))
        elif key == "efficiency.open_loop_horizon_sec":
            norm = max(0.0, 1.0 - min(v / 3.0, 1.0))
        else:
            norm = max(0.0, min(v, 1.0))
        score += w * norm
        weight_sum += w
    if weight_sum <= 0:
        return None
    return round(score / weight_sum, 4)


def build_ablation_compare_report(
    run_reports: List[Dict[str, Any]],
    *,
    experiment_id: str,
    output_dir: Path,
) -> Path:
    """
    消融/对比实验报告：汇总多 run 的指标到一张表。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ablation_compare_{experiment_id}_{ts}.json"
    md_path = output_dir / f"ablation_compare_{experiment_id}_{ts}.md"

    # 收集所有指标名
    all_metric_names = set()
    for r in run_reports:
        all_metric_names.update(r.get("metrics", {}).keys())

    baseline_report = next((r for r in run_reports if r.get("is_baseline")), None)
    if baseline_report is None and run_reports:
        baseline_report = run_reports[0]

    rows = []
    for r in run_reports:
        row = {
            "run_name": r.get("run_name", r.get("test_name", "")),
            "ablation_group": r.get("ablation_group", ""),
            "ablation_description": r.get("ablation_description", ""),
            "ablation_overrides": r.get("ablation_overrides", {}),
            "is_baseline": bool(r.get("is_baseline", False)),
            "test_dir": r.get("test_dir", ""),
            "vla_composite_score": r.get("vla_composite_score"),
        }
        if baseline_report and r.get("vla_composite_score") is not None:
            base_score = baseline_report.get("vla_composite_score")
            if base_score is not None:
                row["delta.vla_composite_score"] = round(
                    float(r["vla_composite_score"]) - float(base_score), 6
                )
        for mn in sorted(all_metric_names):
            md = r.get("metrics", {}).get(mn, {})
            row[mn] = md.get("value") if isinstance(md, dict) else md
            if baseline_report:
                base_md = baseline_report.get("metrics", {}).get(mn, {})
                base_val = base_md.get("value") if isinstance(base_md, dict) else base_md
                if row[mn] is not None and base_val is not None:
                    try:
                        row[f"delta.{mn}"] = round(float(row[mn]) - float(base_val), 6)
                    except (TypeError, ValueError):
                        pass
        rows.append(row)

    payload = {
        "report_type": "ablation_compare",
        "experiment_id": experiment_id,
        "generated_at": datetime.now().astimezone().isoformat(),
        "baseline_run": baseline_report.get("run_name", baseline_report.get("test_name", ""))
        if baseline_report
        else None,
        "runs": run_reports,
        "comparison_table": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    from infer.reports.human_readable import save_ablation_human_reports

    txt_path, csv_path = save_ablation_human_reports(payload, json_path)

    # Markdown 表格（保留，便于 GitHub / 文档查看）
    lines = [
        f"# 消融对比报告: {experiment_id}",
        "",
        f"生成时间: {payload['generated_at']}",
        f"baseline: {payload['baseline_run'] or '-'}",
        "",
        "| group | run | vla_composite | delta | task_success | grasp_success | sync_warn_rate | tracking_err | duration_s |",
        "|-------|-----|---------------|-------|--------------|---------------|----------------|--------------|------------|",
    ]
    for row in rows:
        def _g(key, fmt="{:.3f}"):
            v = row.get(key)
            if v is None:
                return "-"
            try:
                return fmt.format(v)
            except (TypeError, ValueError):
                return str(v)

        lines.append(
            f"| {row.get('ablation_group') or '-'} | {row['run_name']} | "
            f"{_g('vla_composite_score')} | {_g('delta.vla_composite_score')} | "
            f"{_g('task.task_success')} | {_g('task.grasp_success')} | "
            f"{_g('runtime.sync_warning_rate')} | {_g('runtime.tracking_error_mean')} | "
            f"{_g('efficiency.episode_duration_sec', '{:.1f}')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["human_report_txt"] = str(txt_path)
    payload["human_report_csv"] = str(csv_path)
    payload["human_report_md"] = str(md_path)
    return json_path
