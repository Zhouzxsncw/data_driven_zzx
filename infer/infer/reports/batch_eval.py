"""批量评估：CSV 任务标签 + 各 test 自动指标汇总。"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from infer.core.types import MetricContext
from infer.metrics.registry import AUTOMATIC_METRICS, compute_metrics
from infer.reports.single_run import _runtime_composite, _vla_composite

_BOOL_TRUE = {"1", "true", "yes", "y", "是", "成功", "success", "pass", "passed"}
_BOOL_FALSE = {"0", "false", "no", "n", "否", "失败", "fail", "failed"}

LABEL_COLUMNS = (
    "task_success",
    "grasp_success",
    "intention_correct",
    "human_interventions",
    "notes",
)


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in _BOOL_TRUE:
        return True
    if lower in _BOOL_FALSE:
        return False
    raise ValueError(f"无法解析布尔值: {value!r}")


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _normalize_test_name(name: str) -> str:
    text = str(name).strip().replace("\\", "/")
    if "/" in text:
        text = Path(text).name
    if re.fullmatch(r"\d+", text):
        return f"test_{int(text):02d}"
    if re.fullmatch(r"test_?\d+", text, flags=re.I):
        num = int(re.search(r"\d+", text).group())  # type: ignore[union-attr]
        return f"test_{num:02d}"
    return text


def resolve_test_dir(inference_dir: Path, test_ref: str) -> Path:
    inference_dir = inference_dir.resolve()
    ref = str(test_ref).strip()
    if not ref:
        raise ValueError("CSV 中 test 不能为空")
    candidate = Path(ref).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.is_absolute() and candidate.parent.is_dir():
        return candidate.resolve()
    rel = inference_dir / ref
    if rel.is_dir():
        return rel.resolve()
    normalized = _normalize_test_name(ref)
    norm_path = inference_dir / normalized
    if norm_path.is_dir():
        return norm_path.resolve()
    raise FileNotFoundError(f"找不到 test 目录: {ref} (inference_dir={inference_dir})")


def load_labels_csv(path: Path) -> List[Dict[str, Any]]:
    """读取批量标签 CSV。必须含 test 或 test_dir 列。"""
    path = path.expanduser().resolve()
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 无表头: {path}")
        fields = {c.strip().lower(): c for c in reader.fieldnames}
        test_key = fields.get("test") or fields.get("test_dir") or fields.get("test_name")
        if test_key is None:
            raise ValueError("CSV 需包含 test / test_dir / test_name 列")

        for line_no, raw in enumerate(reader, start=2):
            test_ref = (raw.get(test_key) or "").strip()
            if not test_ref:
                continue
            entry: Dict[str, Any] = {"test_ref": test_ref, "test_name": _normalize_test_name(test_ref)}
            for col in LABEL_COLUMNS:
                src = fields.get(col)
                if src is None:
                    continue
                val = raw.get(src)
                if val is None or str(val).strip() == "":
                    continue
                if col == "human_interventions":
                    entry[col] = _parse_int(val)
                elif col == "notes":
                    entry[col] = str(val).strip()
                else:
                    entry[col] = _parse_bool(val)
            entry["_csv_line"] = line_no
            rows.append(entry)
    if not rows:
        raise ValueError(f"CSV 无有效数据行: {path}")
    return rows


def _aggregate_numeric(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "count": float(arr.size),
    }


def _rate(values: Iterable[Optional[bool]]) -> Optional[float]:
    labeled = [1.0 if v else 0.0 for v in values if v is not None]
    if not labeled:
        return None
    return float(np.mean(labeled))


def build_batch_evaluation_report(
    inference_dir: Path,
    labels_csv: Path,
    *,
    experiment_name: str = "",
    policy_type: str = "",
    save: bool = True,
    output_basename: str = "batch_evaluation_report",
) -> Dict[str, Any]:
    """对 inference_dir 下多次 test 生成汇总评估。"""
    inference_dir = inference_dir.resolve()
    label_rows = load_labels_csv(labels_csv)

    per_test: List[Dict[str, Any]] = []
    auto_metric_names = [m.name for m in AUTOMATIC_METRICS]

    for row in label_rows:
        test_dir = resolve_test_dir(inference_dir, row["test_ref"])
        ctx = MetricContext.from_test_dir(test_dir)
        ctx.labels = {k: row[k] for k in LABEL_COLUMNS if k in row and k != "notes"}
        ctx.load()

        auto_metrics = compute_metrics(ctx, ["automatic"])
        task_metrics = compute_metrics(ctx, ["task"])
        merged = {**auto_metrics, **task_metrics}

        runtime_score = _runtime_composite(auto_metrics)
        vla_score = _vla_composite(merged)

        per_test.append(
            {
                "test_name": test_dir.name,
                "test_dir": str(test_dir),
                "csv_line": row.get("_csv_line"),
                "labels": ctx.labels,
                "metrics": {k: v.to_dict() for k, v in merged.items()},
                "runtime_composite_score": runtime_score,
                "vla_composite_score": vla_score,
            }
        )

    # 汇总任务指标（来自 CSV）
    task_summary: Dict[str, Any] = {
        "test_count": len(per_test),
        "task_success_rate": _rate(r.get("labels", {}).get("task_success") for r in per_test),
        "grasp_success_rate": _rate(r.get("labels", {}).get("grasp_success") for r in per_test),
        "intention_correct_rate": _rate(
            r.get("labels", {}).get("intention_correct") for r in per_test
        ),
        "human_interventions_total": sum(
            int(r.get("labels", {}).get("human_interventions") or 0) for r in per_test
        ),
    }

    # 汇总自动指标（跨 test 均值）
    auto_summary: Dict[str, Any] = {}
    for name in auto_metric_names:
        vals = []
        for r in per_test:
            md = r.get("metrics", {}).get(name)
            if isinstance(md, dict) and md.get("value") is not None:
                vals.append(float(md["value"]))
        if vals:
            auto_summary[name] = _aggregate_numeric(vals)

    runtime_scores = [r["runtime_composite_score"] for r in per_test if r["runtime_composite_score"] is not None]
    vla_scores = [r["vla_composite_score"] for r in per_test if r["vla_composite_score"] is not None]

    report: Dict[str, Any] = {
        "report_type": "batch_evaluation",
        "generated_at": datetime.now().astimezone().isoformat(),
        "experiment_name": experiment_name,
        "policy_type": policy_type,
        "inference_dir": str(inference_dir),
        "labels_csv": str(labels_csv.resolve()),
        "task_summary": task_summary,
        "automatic_metrics_summary": auto_summary,
        "runtime_composite_summary": _aggregate_numeric(runtime_scores) if runtime_scores else {},
        "vla_composite_summary": _aggregate_numeric(vla_scores) if vla_scores else {},
        "tests": per_test,
    }

    if save:
        json_path = inference_dir / f"{output_basename}.json"
        txt_path = inference_dir / f"{output_basename}.txt"
        csv_path = inference_dir / f"{output_basename}.csv"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        txt_path.write_text(_format_batch_txt(report), encoding="utf-8")
        _write_batch_csv(report, csv_path)
        report["output_json"] = str(json_path)
        report["output_txt"] = str(txt_path)
        report["output_csv"] = str(csv_path)

    return report


def _format_batch_txt(report: Dict[str, Any]) -> str:
    lines = ["=" * 72, "VLA 批量评估报告", "=" * 72]
    lines.append(f"生成时间: {report.get('generated_at', '-')}")
    lines.append(f"实验: {report.get('experiment_name') or '-'}")
    lines.append(f"推理目录: {report.get('inference_dir', '-')}")
    lines.append(f"标签 CSV: {report.get('labels_csv', '-')}")
    lines.append("")

    ts = report.get("task_summary") or {}
    lines.append("【任务汇总 — 来自 CSV 标签】")
    lines.append(f"  测试次数: {ts.get('test_count', 0)}")
    for key, label in (
        ("task_success_rate", "任务成功率 (TSR)"),
        ("grasp_success_rate", "抓取成功率"),
        ("intention_correct_rate", "意图正确率"),
    ):
        v = ts.get(key)
        lines.append(f"  {label}: {v * 100:.1f}%" if v is not None else f"  {label}: 未标注")
    lines.append(f"  人工接管总次数: {ts.get('human_interventions_total', 0)}")
    lines.append("")

    vla = (report.get("vla_composite_summary") or {}).get("mean")
    runtime = (report.get("runtime_composite_summary") or {}).get("mean")
    lines.append("【综合分汇总】")
    if vla is not None:
        lines.append(f"  VLA 综合分（含任务）均值: {vla:.3f}")
    if runtime is not None:
        lines.append(f"  Runtime 综合分（仅自动指标）均值: {runtime:.3f}")
    lines.append("")

    auto = report.get("automatic_metrics_summary") or {}
    if auto:
        lines.append("【自动指标跨 test 均值】")
        for name, stats in sorted(auto.items()):
            lines.append(f"  {name}: mean={stats.get('mean', 0):.4g}, p95={stats.get('p95', 0):.4g}, n={int(stats.get('count', 0))}")
        lines.append("")

    lines.append("【各 test 明细】")
    for r in report.get("tests") or []:
        labels = r.get("labels") or {}
        ts_ok = labels.get("task_success")
        ts_text = "成功" if ts_ok else ("失败" if ts_ok is False else "-")
        runtime_s = r.get("runtime_composite_score")
        vla_s = r.get("vla_composite_score")
        lines.append(
            f"  {r.get('test_name')}: task={ts_text}, "
            f"runtime={runtime_s:.3f if runtime_s is not None else 'n/a'}, "
            f"vla={vla_s:.3f if vla_s is not None else 'n/a'}"
        )
    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def _write_batch_csv(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tests = report.get("tests") or []
    auto_names = sorted((report.get("automatic_metrics_summary") or {}).keys())
    fieldnames = [
        "test_name",
        "test_dir",
        "task_success",
        "grasp_success",
        "intention_correct",
        "human_interventions",
        "runtime_composite_score",
        "vla_composite_score",
        *auto_names,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in tests:
            labels = r.get("labels") or {}
            row = {
                "test_name": r.get("test_name"),
                "test_dir": r.get("test_dir"),
                "task_success": labels.get("task_success"),
                "grasp_success": labels.get("grasp_success"),
                "intention_correct": labels.get("intention_correct"),
                "human_interventions": labels.get("human_interventions"),
                "runtime_composite_score": r.get("runtime_composite_score"),
                "vla_composite_score": r.get("vla_composite_score"),
            }
            metrics = r.get("metrics") or {}
            for name in auto_names:
                md = metrics.get(name)
                row[name] = md.get("value") if isinstance(md, dict) else None
            writer.writerow(row)
