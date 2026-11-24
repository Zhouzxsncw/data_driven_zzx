#!/usr/bin/env python
"""为本地 LeRobot 数据集 meta/stats.json 补全 q01/q99（PI05 需要；不改 src/lerobot）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def augment(stats: dict) -> dict:
    for key, feat in stats.items():
        if not isinstance(feat, dict) or "min" not in feat or "max" not in feat:
            continue
        if "q01" in feat and "q99" in feat:
            continue
        feat["q01"] = feat["min"]
        feat["q99"] = feat["max"]
        if "q10" not in feat and "mean" in feat:
            feat["q10"] = feat["mean"]
        if "q50" not in feat and "mean" in feat:
            feat["q50"] = feat["mean"]
        if "q90" not in feat and "mean" in feat:
            feat["q90"] = feat["mean"]
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    path = args.root / "meta" / "stats.json"
    stats = json.loads(path.read_text())
    stats = augment(stats)
    path.write_text(json.dumps(stats, indent=4) + "\n")
    print(f"Updated quantile fields in {path}")


if __name__ == "__main__":
    main()
