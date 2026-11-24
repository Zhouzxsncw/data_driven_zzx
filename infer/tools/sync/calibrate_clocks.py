#!/usr/bin/env python3
"""
从 machine_flow.jsonl 估算 state / camera 时钟偏移（离线工具，不修改推理行为）。

用法:
  python3 tools/sync/calibrate_clocks.py \\
      --machine-flow path/to/machine_flow.jsonl \\
      --output configs/sync_calibration.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))


def _load_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def estimate_offsets(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """用 chunk_inference 的 state/image 时间戳差估算 camera 相对 robot 的偏移。"""
    diffs: List[float] = []
    for e in events:
        et = e.get("type") or e.get("event")
        if et != "chunk_inference":
            continue
        p = e.get("payload") or e
        st = p.get("state_timestamp")
        it = p.get("image_timestamp")
        if isinstance(st, (int, float)) and isinstance(it, (int, float)):
            diffs.append(float(st) - float(it))

    if not diffs:
        return {"robot_state": 0.0, "camera": 0.0}

    # 中位数 state - image：若 camera 偏晚，camera offset 应 +median
    median_diff = statistics.median(diffs)
    return {
        "robot_state": 0.0,
        "camera": round(median_diff, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="从 machine_flow 估算 clock_offset")
    parser.add_argument("--machine-flow", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO / "configs" / "sync_calibration.yaml",
        help="输出 YAML（默认 configs/sync_calibration.yaml）",
    )
    args = parser.parse_args()

    if not args.machine_flow.is_file():
        raise SystemExit(f"文件不存在: {args.machine_flow}")

    events = _load_events(args.machine_flow)
    offsets = estimate_offsets(events)
    payload = {
        "clock_offset": offsets,
        "_note": f"由 {args.machine_flow.name} 估算，样本 chunk 数见终端输出",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.dump({"clock_offset": offsets}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"[OK] 写入 {args.output}")
    print(f"     clock_offset: {offsets}")
    print("     请在真机小批量验证后再用于生产配置。")


if __name__ == "__main__":
    main()
