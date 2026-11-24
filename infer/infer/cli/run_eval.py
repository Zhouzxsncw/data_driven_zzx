#!/usr/bin/env python3
"""离线评估：对已有 test 目录生成 evaluation_report.json。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_INFER_PKG = Path(__file__).resolve().parents[1]
_REPO = _INFER_PKG.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from infer.logging_setup import setup_logging
from infer.metrics.registry import list_metric_presets, list_metrics
from infer.reports.single_run import build_single_run_report


def main() -> None:
    p = argparse.ArgumentParser(description="离线推理评估")
    p.add_argument("--test-dir", type=Path, default=None, help="组目录（如 action-step_10/ 或 ood-language/）")
    p.add_argument("--labels", type=Path, default=None, help="labels.json（TSR/grasp 等）")
    p.add_argument("--list-metrics", action="store_true")
    p.add_argument("--metrics", nargs="*", default=["automatic"], help="指标 preset: automatic / task / all")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    setup_logging("INFO")

    if args.list_metrics:
        print("指标 preset:")
        for preset in list_metric_presets():
            print(f"  - {preset}")
        print("可用指标:")
        for m in list_metrics():
            print(f"  - {m}")
        return

    if args.test_dir is None:
        p.error("--test-dir 为必填参数（除非使用 --list-metrics）")

    test_dir = args.test_dir.resolve()
    report = build_single_run_report(
        test_dir,
        metric_names=args.metrics if args.metrics != ["all"] else None,
        labels_file=args.labels,
        save=True,
    )
    score = report.get("runtime_composite_score", report.get("vla_composite_score"))
    print(f"Runtime 综合分: {score}")
    print(f"[OK] JSON: {test_dir / 'evaluation_report.json'}")
    print(f"[OK] TXT:  {test_dir / 'evaluation_report.txt'}")
    print(f"[OK] CSV:  {test_dir / 'evaluation_report.csv'}")


if __name__ == "__main__":
    main()
