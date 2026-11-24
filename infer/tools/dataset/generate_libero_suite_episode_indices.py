#!/usr/bin/env python3
"""从 data/lerobot/libero 合并集生成各套件 episode 索引（供训练过滤）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _normalize_task(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _suite_task_languages(suite_name: str) -> set[str]:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    return {_normalize_task(t.language) for t in suite.tasks}


def episode_indices_for_suite(combined_root: Path, suite_name: str) -> list[int]:
    import pandas as pd

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("lerobot/libero", root=combined_root)
    tasks_df = pd.read_parquet(combined_root / "meta/tasks.parquet")
    suite_langs = _suite_task_languages(suite_name)
    suite_task_ids = {
        int(row["task_index"])
        for task_text, row in tasks_df.iterrows()
        if _normalize_task(str(task_text)) in suite_langs
    }

    ep_to_task: dict[int, int] = {}
    for parquet_path in sorted((combined_root / "data").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        for ep_idx, grp in df.groupby("episode_index"):
            if int(ep_idx) not in ep_to_task:
                ep_to_task[int(ep_idx)] = int(grp["task_index"].iloc[0])

    return sorted(ep for ep, tid in ep_to_task.items() if tid in suite_task_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined-root",
        type=Path,
        default=None,
        help="合并数据集目录，默认 <repo>/data/lerobot/libero",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="索引输出目录，默认 <repo>/data/lerobot/_suite_indices",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    combined_root = (args.combined_root or repo_root / "data/lerobot/libero").resolve()
    output_dir = (args.output_dir or repo_root / "data/lerobot/_suite_indices").resolve()

    if not (combined_root / "meta/info.json").is_file():
        print(f"错误: 未找到合并集 {combined_root}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(repo_root / "src"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for suite in SUITES:
        indices = episode_indices_for_suite(combined_root, suite)
        out_txt = output_dir / f"{suite}.txt"
        out_json = output_dir / f"{suite}.json"
        out_txt.write_text(",".join(str(i) for i in indices), encoding="utf-8")
        out_json.write_text(
            json.dumps(
                {
                    "suite": suite,
                    "combined_root": str(combined_root),
                    "repo_id": "lerobot/libero",
                    "episodes": indices,
                    "count": len(indices),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{suite}: {len(indices)} episodes -> {out_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
