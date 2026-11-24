"""从本地 LeRobot 数据集加载归一化统计（离线，不访问 Hugging Face）。

ACT / DP adapter 在 load() 时读取 dataset_root/meta/stats.json，
供 make_pre_post_processors 做 state/action 归一化与反归一化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from lerobot.datasets.utils import load_stats


def load_local_dataset_stats(dataset_root: Path) -> Dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root 不存在: {root}")
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"未找到 {stats_path}；请确认 dataset_root 指向含 meta/ 的数据集目录"
        )
    stats = load_stats(root)
    if stats is None:
        raise FileNotFoundError(f"无法解析 {stats_path}")
    return stats
