#!/usr/bin/env python3
"""批量评估：CSV 记录各 test 任务结果 + 自动指标汇总。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_INFER_PKG = Path(__file__).resolve().parents[1]
_REPO = _INFER_PKG.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from infer.logging_setup import setup_logging
from infer.reports.batch_eval import build_batch_evaluation_report


def main() -> None:
    p = argparse.ArgumentParser(
        description="批量评估：inference/ 下多次 test + CSV 任务标签 → 汇总报告"
    )
    p.add_argument(
        "--inference-dir",
        type=Path,
        required=True,
        help="批量结果根目录（含 action-step_* / ood-language/ 组目录）",
    )
    p.add_argument(
        "--labels-csv",
        type=Path,
        required=True,
        help="各 test 任务结果 CSV（见 config/labels_batch_example.csv）",
    )
    p.add_argument("--experiment-name", type=str, default="")
    p.add_argument("--policy-type", type=str, default="")
    p.add_argument("--output-basename", type=str, default="batch_evaluation_report")
    args = p.parse_args()

    setup_logging("INFO")

    report = build_batch_evaluation_report(
        args.inference_dir.resolve(),
        args.labels_csv.resolve(),
        experiment_name=args.experiment_name,
        policy_type=args.policy_type,
        save=True,
        output_basename=args.output_basename,
    )

    ts = report.get("task_summary") or {}
    tsr = ts.get("task_success_rate")
    print(f"测试次数: {ts.get('test_count', 0)}")
    if tsr is not None:
        print(f"任务成功率 (TSR): {tsr * 100:.1f}%")
    runtime_mean = (report.get("runtime_composite_summary") or {}).get("mean")
    vla_mean = (report.get("vla_composite_summary") or {}).get("mean")
    if runtime_mean is not None:
        print(f"Runtime 综合分均值: {runtime_mean:.3f}")
    if vla_mean is not None:
        print(f"VLA 综合分均值: {vla_mean:.3f}")
    print(f"[OK] JSON: {report.get('output_json')}")
    print(f"[OK] TXT:  {report.get('output_txt')}")
    print(f"[OK] CSV:  {report.get('output_csv')}")


if __name__ == "__main__":
    main()
