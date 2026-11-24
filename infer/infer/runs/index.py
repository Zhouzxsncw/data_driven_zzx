"""推理产物目录校验（扁平布局；兼容旧版 test_XX 子目录）。"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

# 本组说明文件，推理开始前就会写上，不能当成上一次的推理结果。
_PRE_RUN_FILES = {"ood_group_meta.json", "ood_preset_meta.json"}
_RUN_DIR_NAME = re.compile(r"^run_(\d{8}_\d{6})(?:_(\d+))?$")
_EXP_DIR_NAME = re.compile(r"^exp(\d+)$", re.IGNORECASE)


def run_dir_has_valid_trajectory(run_dir: Path) -> bool:
    traj = run_dir / "trajectory_data" / "trajectory.parquet"
    return traj.is_file() and traj.stat().st_size > 0


def run_dir_has_valid_videos(run_dir: Path) -> bool:
    videos_dir = run_dir / "videos"
    if not videos_dir.is_dir():
        return False
    for pattern in ("*.mp4", "*.avi"):
        for video_path in videos_dir.glob(pattern):
            if video_path.is_file() and video_path.stat().st_size > 0:
                return True
    return False


def run_dir_is_complete(run_dir: Path) -> bool:
    return (run_dir / "inference_summary.json").is_file()


def run_dir_is_recording_incomplete(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    if run_dir_is_complete(run_dir):
        return False
    return not run_dir_has_valid_trajectory(run_dir) or not run_dir_has_valid_videos(run_dir)


def remove_incomplete_run_dir(run_dir: Path) -> bool:
    """不再自动删除。旧推理结果保留，调用方改走 allocate_run_dir。"""
    return False


def run_dir_has_kept_results(run_dir: Path) -> bool:
    """目录里是否已经有上次推理留下的文件。空的 videos/、trajectory_data/ 不算。"""
    if not run_dir.is_dir():
        return False
    for child in run_dir.iterdir():
        if child.name.startswith(".") or child.name in _PRE_RUN_FILES:
            continue
        if child.is_dir() and child.name in {"videos", "trajectory_data"}:
            if any(child.iterdir()):
                return True
            continue
        return True
    return False


def _run_dir_sort_key(path: Path) -> tuple:
    match = _RUN_DIR_NAME.match(path.name)
    if match is None:
        return ("", 0)
    return (match.group(1), int(match.group(2) or 1))


def _exp_dir_index(path: Path) -> Optional[int]:
    match = _EXP_DIR_NAME.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def list_exp_dirs(output_dir: Path) -> list[Path]:
    """action-step / 组目录下已有的 exp1、exp2…（按编号排序；兼容旧 EXP）。"""
    if not output_dir.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        index = _exp_dir_index(child)
        if index is None:
            continue
        found.append((index, child))
    found.sort(key=lambda item: item[0])
    return [path for _, path in found]


def next_exp_index(output_dir: Path) -> int:
    """下一个可用的 exp 编号。已有 exp3 则返回 4；没有则从 1 起。"""
    indices = [_exp_dir_index(path) for path in list_exp_dirs(output_dir)]
    indices = [i for i in indices if i is not None]
    if not indices:
        return 1
    return max(indices) + 1


def latest_saved_run_dir(output_dir: Path) -> Path:
    """最近一次真正写过文件的 exp* 或 run_时间 目录。没有则返回 output_dir 本身。"""
    if not output_dir.is_dir():
        return output_dir
    exps = [
        child
        for child in list_exp_dirs(output_dir)
        if run_dir_has_kept_results(child)
    ]
    if exps:
        return exps[-1]
    runs = [
        child
        for child in output_dir.iterdir()
        if child.is_dir()
        and _RUN_DIR_NAME.match(child.name)
        and run_dir_has_kept_results(child)
    ]
    if not runs:
        return output_dir
    runs.sort(key=_run_dir_sort_key)
    return runs[-1]


def allocate_run_dir(output_dir: Path) -> Path:
    """已有产物时在其下新建 run_时间戳/，不改动旧文件。空目录仍直接使用。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not run_dir_has_kept_results(output_dir):
        return output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"run_{stamp}"
    n = 2
    while candidate.exists():
        candidate = output_dir / f"run_{stamp}_{n}"
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def allocate_exp_dir(output_dir: Path) -> Path:
    """批量 action-step：在参数目录下新建 exp1、exp2…，不改动旧文件。

    ``action-step_10`` 只表示 n_action_steps=10。每次真机测试单独一份 exp。
    父目录里旧的扁平结果、旧 EXP* / exp* 或 run_* 都保留。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n = next_exp_index(output_dir)
    candidate = output_dir / f"exp{n}"
    while candidate.exists():
        n += 1
        candidate = output_dir / f"exp{n}"
    candidate.mkdir(parents=True)
    return candidate


def clear_run_dir_contents(run_dir: Path) -> None:
    """删除目录内全部内容（保留目录本身）。"""
    if not run_dir.is_dir():
        run_dir.mkdir(parents=True, exist_ok=True)
        return
    for child in run_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def legacy_test_subdirs(output_dir: Path) -> list[Path]:
    return sorted(
        p for p in output_dir.glob("test_*") if p.is_dir() and p.name[5:].isdigit()
    )


def resolve_run_dir(output_dir: Path) -> Path:
    """
    返回实际持有产物的目录：优先扁平布局，否则兼容旧版 test_XX/。
    """
    if run_dir_is_complete(output_dir) or run_dir_is_recording_incomplete(output_dir):
        return output_dir
    legacy = legacy_test_subdirs(output_dir)
    if legacy:
        return legacy[-1]
    return output_dir


# 兼容旧名（勿用 test_ 前缀，否则 pytest 会误收集为测试用例）
is_test_dir_recording_incomplete = run_dir_is_recording_incomplete
remove_incomplete_test_dir = remove_incomplete_run_dir


def resolve_next_test_index(inference_dir: Path, *, record: bool = True) -> int:
    """已废弃序号分配：扁平布局下恒为 1，仅保留 API 兼容。不删除已有目录。"""
    del record, inference_dir
    return 1
