#!/usr/bin/env python3
"""将 lerobot/libero 合并包拆成各套件目录（data/lerobot/libero_spatial 等）。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 与 train/scripts 中 SIM_ENV_TASK 目录名一致
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _normalize_task(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _suite_task_languages(suite_name: str) -> set[str]:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    return {_normalize_task(t.language) for t in suite.tasks}


def _episode_task_indices(dataset) -> dict[int, int]:
    import pandas as pd

    from lerobot.datasets.utils import load_episodes

    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    ep_to_task: dict[int, int] = {}
    data_dir = dataset.root / "data"
    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        for ep_idx, grp in df.groupby("episode_index"):
            if int(ep_idx) not in ep_to_task:
                ep_to_task[int(ep_idx)] = int(grp["task_index"].iloc[0])
    return ep_to_task


def _episode_indices_for_suite(dataset, suite_name: str) -> list[int]:
    import pandas as pd

    tasks_path = dataset.root / "meta/tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"未找到 {tasks_path}")

    tasks_df = pd.read_parquet(tasks_path)
    suite_langs = _suite_task_languages(suite_name)
    suite_task_ids = set()
    for task_text, row in tasks_df.iterrows():
        if _normalize_task(str(task_text)) in suite_langs:
            suite_task_ids.add(int(row["task_index"]))

    ep_to_task = _episode_task_indices(dataset)
    return sorted(ep for ep, tid in ep_to_task.items() if tid in suite_task_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="合并数据集根目录，例如 data/lerobot/libero",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出父目录，例如 data/lerobot",
    )
    parser.add_argument(
        "--suites",
        nargs="*",
        default=list(SUITES),
        choices=SUITES,
        help="要生成的套件（默认四个 10 任务套件）",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="目标目录已有 meta/info.json 时跳过",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))

    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    if not (source_root / "meta/info.json").is_file():
        logger.error("源数据集不完整: %s", source_root)
        return 1

    dataset = LeRobotDataset("lerobot/libero", root=source_root)
    logger.info(
        "源集: %s episodes, %s frames, %s tasks",
        dataset.meta.total_episodes,
        dataset.meta.total_frames,
        dataset.meta.total_tasks,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 0

    for suite in args.suites:
        dest = output_dir / suite
        if args.skip_existing and (dest / "meta/info.json").is_file():
            logger.info("[SKIP] %s 已存在", dest)
            continue

        episodes = _episode_indices_for_suite(dataset, suite)
        if not episodes:
            logger.error("套件 %s 未匹配到任何 episode", suite)
            exit_code = 1
            continue

        logger.info("拆分 %s: %d episodes -> %s", suite, len(episodes), dest)
        if dest.exists():
            import shutil

            shutil.rmtree(dest)

        split_dataset(
            dataset,
            splits={suite: episodes},
            output_dir=output_dir,
        )
        out_ds = LeRobotDataset(f"lerobot/{suite}", root=dest)
        logger.info(
            "[OK] %s: %d episodes, %d frames",
            suite,
            out_ds.meta.total_episodes,
            out_ds.meta.total_frames,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
