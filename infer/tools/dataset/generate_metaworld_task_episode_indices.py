#!/usr/bin/env python3
"""从 lerobot/metaworld 合并集按 MT50 task_id 生成各单任务 episode 索引。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def episode_indices_for_task_id(metaworld_root: Path, task_id: int) -> list[int]:
    import pandas as pd

    ep_to_tid: dict[int, int] = {}
    for parquet_path in sorted((metaworld_root / "data").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index", "task_id"])
        for ep_idx, grp in df.groupby("episode_index"):
            if int(ep_idx) not in ep_to_tid:
                ep_to_tid[int(ep_idx)] = int(grp["task_id"].iloc[0])

    return sorted(ep for ep, tid in ep_to_tid.items() if tid == task_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metaworld-root",
        type=Path,
        default=None,
        help="MetaWorld 数据集目录，默认 <repo>/data/lerobot/metaworld",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="索引输出目录，默认 <repo>/data/lerobot/_metaworld_indices",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    metaworld_root = (args.metaworld_root or repo_root / "data/lerobot/metaworld").resolve()
    output_dir = (args.output_dir or repo_root / "data/lerobot/_metaworld_indices").resolve()

    if not (metaworld_root / "meta/info.json").is_file():
        print(f"错误: 未找到 MetaWorld 数据集 {metaworld_root}", file=sys.stderr)
        return 1

    cfg_path = repo_root / "src/lerobot/envs/metaworld_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    task_name_to_id: dict[str, int] = cfg["TASK_NAME_TO_ID"]

    output_dir.mkdir(parents=True, exist_ok=True)
    for task_key, task_id in sorted(task_name_to_id.items(), key=lambda x: x[1]):
        indices = episode_indices_for_task_id(metaworld_root, task_id)
        out_txt = output_dir / f"{task_key}.txt"
        out_json = output_dir / f"{task_key}.json"
        out_txt.write_text(",".join(str(i) for i in indices), encoding="utf-8")
        out_json.write_text(
            json.dumps(
                {
                    "task_key": task_key,
                    "task_id": task_id,
                    "metaworld_root": str(metaworld_root),
                    "repo_id": "lerobot/metaworld",
                    "episodes": indices,
                    "count": len(indices),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{task_key} (task_id={task_id}): {len(indices)} episodes -> {out_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
