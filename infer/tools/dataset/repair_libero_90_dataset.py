#!/usr/bin/env python
"""修复 data/lerobot/libero_90：补 meta、生成 episodes_stats、v2→v3 转换、特征名对齐 LeRobot/LIBERO。"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import io
import jsonlines
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train" / "src"))

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, get_feature_stats
from lerobot.datasets.utils import INFO_PATH, load_info, serialize_dict, write_info
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    V21,
    convert_data,
    convert_episodes_metadata,
    convert_info,
    convert_tasks,
    convert_videos,
    legacy_load_episodes,
)

HF_REPO = "jesbu1/libero_90_lerobot"
FEATURE_RENAME = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.image2",
    "state": "observation.state",
    "actions": "action",
}

log = logging.getLogger(__name__)


EXPECTED_EPISODES = 3959


def dataset_root(data_root: Path) -> Path:
    return data_root / "lerobot" / "libero_90"


def list_episode_parquets(root: Path) -> list[Path]:
    paths = [
        p for p in (root / "data").glob("*/*.parquet") if p.name.startswith("episode_")
    ]
    return sorted(paths, key=episode_index_from_path)


def is_v3_complete(root: Path, expected: int = EXPECTED_EPISODES) -> bool:
    if not (root / "meta" / "info.json").is_file():
        return False
    info = load_info(root)
    if info.get("codebase_version") != "v3.0":
        return False
    n_eps = int(info.get("total_episodes", 0))
    if n_eps < expected:
        return False
    tasks = root / "meta/tasks.parquet"
    if not tasks.is_file() or len(pd.read_parquet(tasks)) == 0:
        return False
    return len(list((root / "data").glob("*/*.parquet"))) > 0


def validate_loadable(root: Path) -> None:
    import sys

    tools = ROOT / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import lerobot_env_bootstrap

    lerobot_env_bootstrap.bootstrap()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(HF_REPO, root=root, revision="v3.0")
    sample = ds[0]
    assert "task" in sample and "action" in sample
    log.info("加载校验通过: len=%d sample_keys=%d", len(ds), len(sample))


def count_episode_parquets(root: Path) -> int:
    return len(list_episode_parquets(root))


def download_meta(root: Path, hf_bin: Path) -> None:
    if (root / "meta" / "info.json").is_file():
        log.info("meta/ 已存在，跳过下载")
        return
    env = os.environ.copy()
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    subprocess.run(
        [
            str(hf_bin),
            "download",
            HF_REPO,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(root),
            "--include",
            "meta/*",
        ],
        check=True,
        env=env,
    )


def resume_data_download(
    root: Path,
    hf_bin: Path,
    expected: int = EXPECTED_EPISODES,
    max_attempts: int = 100,
) -> None:
    env = os.environ.copy()
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "600")

    for attempt in range(1, max_attempts + 1):
        n_local = count_episode_parquets(root)
        if n_local >= expected:
            log.info("下载完成: %d/%d episodes", n_local, expected)
            return
        log.info("续传 data/* 第 %d 次 (已有 %d/%d)...", attempt, n_local, expected)
        try:
            subprocess.run(
                [
                    str(hf_bin),
                    "download",
                    HF_REPO,
                    "--repo-type",
                    "dataset",
                    "--local-dir",
                    str(root),
                ],
                check=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            log.warning("下载中断 (attempt %d): %s", attempt, exc)
            time.sleep(min(30, 5 * attempt))
            continue
        n_after = count_episode_parquets(root)
        if n_after >= expected:
            log.info("下载完成: %d/%d episodes", n_after, expected)
            return
        if n_after <= n_local:
            log.warning("本轮无新增 episode (%d)，等待后重试", n_after)
            time.sleep(min(60, 10 * attempt))

    n_final = count_episode_parquets(root)
    raise SystemExit(f"下载未完成: {n_final}/{expected} episodes（已重试 {max_attempts} 次）")


def episode_index_from_path(path: Path) -> int:
    m = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not m:
        raise ValueError(f"无法解析 episode 索引: {path}")
    return int(m.group(1))


def _image_dicts_to_numpy(col) -> np.ndarray:
    """HF Image 列（dict bytes/path）-> (T, C, H, W) float-ready uint8 CHW。"""
    frames = []
    for item in col:
        if isinstance(item, dict) and item.get("bytes"):
            img = PILImage.open(io.BytesIO(item["bytes"])).convert("RGB")
        elif isinstance(item, dict) and item.get("path"):
            img = PILImage.open(item["path"]).convert("RGB")
        else:
            img = PILImage.open(item).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        frames.append(np.transpose(arr, (2, 0, 1)))
    return np.stack(frames, axis=0)


def _episode_data_for_stats(df: pd.DataFrame, features: dict) -> dict:
    ep_data = {}
    for col in df.columns:
        if col not in features:
            continue
        if features[col]["dtype"] in ("image", "video"):
            ep_data[col] = _image_dicts_to_numpy(df[col].tolist())
        else:
            col_vals = df[col].tolist()
            if col_vals and isinstance(col_vals[0], np.ndarray):
                ep_data[col] = np.stack([np.asarray(v, dtype=np.float32) for v in col_vals])
            else:
                ep_data[col] = df[col].to_numpy(dtype=np.float32)
    return ep_data


def generate_episodes_stats(
    root: Path, max_episodes: int | None = None, episode_paths: list[Path] | None = None
) -> None:
    stats_path = root / "meta" / "episodes_stats.jsonl"
    n_target = count_episode_parquets(root)
    if max_episodes is not None:
        n_target = min(n_target, max_episodes)
    if stats_path.is_file() and max_episodes is None:
        with jsonlines.open(stats_path, "r") as reader:
            n_existing = sum(1 for _ in reader)
        if n_existing == n_target:
            log.info("episodes_stats.jsonl 已存在 (%d)", n_existing)
            return
        log.warning("episodes_stats 行数 %d != 本地 %d，重新生成", n_existing, n_target)

    info = load_info(root)
    features = info["features"]
    ep_paths = episode_paths if episode_paths is not None else list_episode_parquets(root)
    if max_episodes is not None:
        ep_paths = ep_paths[:max_episodes]

    rows: list[dict] = []
    done_indices: set[int] = set()
    if stats_path.is_file() and max_episodes is None:
        with jsonlines.open(stats_path, "r") as reader:
            for row in reader:
                rows.append(row)
                done_indices.add(row["episode_index"])
        if done_indices:
            log.info("续写 episodes_stats：已完成 %d 条", len(done_indices))

    for ep_path in tqdm(ep_paths, desc="episodes_stats"):
        ep_idx = episode_index_from_path(ep_path)
        if ep_idx in done_indices:
            continue
        df = pd.read_parquet(ep_path)
        ep_data = _episode_data_for_stats(df, features)
        stats = {}
        for key, data in ep_data.items():
            if features[key]["dtype"] == "string":
                continue
            if features[key]["dtype"] in ("image", "video"):
                axes_to_reduce = (0, 2, 3)
                keepdims = True
            else:
                axes_to_reduce = 0
                keepdims = data.ndim == 1
            stats[key] = get_feature_stats(
                data, axis=axes_to_reduce, keepdims=keepdims, quantile_list=DEFAULT_QUANTILES
            )
            if features[key]["dtype"] in ("image", "video"):
                stats[key] = {
                    k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                    for k, v in stats[key].items()
                }
        rows.append({"episode_index": ep_idx, "stats": serialize_dict(stats)})
        done_indices.add(ep_idx)
        if len(rows) % 50 == 0:
            with jsonlines.open(stats_path, mode="w") as writer:
                writer.write_all(rows)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(stats_path, mode="w") as writer:
        writer.write_all(rows)
    log.info("已写入 %s (%d episodes)", stats_path, len(rows))


def patch_info_for_conversion(root: Path) -> None:
    info_path = root / INFO_PATH
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") == "v2.0":
        info["codebase_version"] = V21
        info_path.write_text(json.dumps(info, indent=4) + "\n")
        log.info("已将 codebase_version v2.0 标记为 %s 以供转换", V21)


def convert_data_subset(root: Path, new_root: Path, episode_paths: list[Path], data_file_size_in_mb: int):
    """仅转换指定 episode parquet 列表（避免下载进行中数量漂移）。"""
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
        concat_data_files,
        get_image_keys,
        get_parquet_file_size_in_mb,
        get_parquet_num_frames,
        update_chunk_file_indices,
    )
    from lerobot.datasets.utils import DEFAULT_CHUNK_SIZE

    image_keys = get_image_keys(root)
    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    num_frames = 0
    paths_to_cat = []
    episodes_metadata = []

    for ep_path in tqdm(episode_paths, desc="convert data files"):
        ep_size_in_mb = get_parquet_file_size_in_mb(ep_path)
        ep_num_frames = get_parquet_num_frames(ep_path)
        ep_metadata = {
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": num_frames,
            "dataset_to_index": num_frames + ep_num_frames,
        }
        size_in_mb += ep_size_in_mb
        num_frames += ep_num_frames
        episodes_metadata.append(ep_metadata)
        ep_idx += 1

        if size_in_mb < data_file_size_in_mb:
            paths_to_cat.append(ep_path)
            continue

        concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
        size_in_mb = 0
        paths_to_cat = []
        paths_to_cat.append(ep_path)
        size_in_mb = ep_size_in_mb

    if paths_to_cat:
        concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)

    return episodes_metadata


def convert_to_v30(root: Path, episode_paths: list[Path] | None = None) -> None:
    info = load_info(root)
    if info.get("codebase_version") == "v3.0":
        log.info("已是 v3.0，跳过转换")
        return

    tasks_jsonl_backup = (root / "meta/tasks.jsonl").read_bytes() if (root / "meta/tasks.jsonl").is_file() else None

    old_root = root.parent / f"{root.name}_old"
    new_root = root.parent / f"{root.name}_v30"
    if old_root.is_dir() and root.is_dir():
        shutil.rmtree(root)
        shutil.move(str(old_root), str(root))
    if new_root.is_dir():
        shutil.rmtree(new_root)

    data_file_size_in_mb = info.get("data_files_size_in_mb", 100)
    video_file_size_in_mb = info.get("video_files_size_in_mb", 500)

    ep_paths = episode_paths if episode_paths is not None else list_episode_parquets(root)

    convert_info(root, new_root, data_file_size_in_mb, video_file_size_in_mb)
    convert_tasks(root, new_root)
    episodes_metadata = convert_data_subset(root, new_root, ep_paths, data_file_size_in_mb)
    episodes_videos_metadata = convert_videos(root, new_root, video_file_size_in_mb)
    convert_episodes_metadata(root, new_root, episodes_metadata, episodes_videos_metadata)

    shutil.move(str(root), str(old_root))
    shutil.move(str(new_root), str(root))
    shutil.rmtree(old_root, ignore_errors=True)
    if tasks_jsonl_backup is not None and not (root / "meta/tasks.jsonl").is_file():
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "meta/tasks.jsonl").write_bytes(tasks_jsonl_backup)
    log.info("v3.0 转换完成: %s", root)


def ensure_tasks_parquet(root: Path, hf_bin: Path) -> None:
    """convert 后若 tasks.parquet 为空，从 Hub meta/tasks.jsonl 恢复。"""
    tasks_path = root / "meta/tasks.parquet"
    if tasks_path.is_file():
        df = pd.read_parquet(tasks_path)
        if len(df) > 0:
            return
    legacy = root / "meta/tasks.jsonl"
    if not legacy.is_file():
        env = os.environ.copy()
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
        subprocess.run(
            [
                str(hf_bin),
                "download",
                HF_REPO,
                "meta/tasks.jsonl",
                "--repo-type",
                "dataset",
                "--local-dir",
                str(root),
            ],
            check=True,
            env=env,
        )
    if not legacy.is_file():
        raise FileNotFoundError(f"无法恢复 tasks: {legacy}")
    tasks = {item["task_index"]: item["task"] for item in jsonlines.open(legacy)}
    df_tasks = pd.DataFrame({"task_index": list(tasks.keys())}, index=list(tasks.values()))
    from lerobot.datasets.utils import write_tasks

    write_tasks(df_tasks, root)
    log.info("已写入 meta/tasks.parquet (%d tasks)", len(df_tasks))


def rename_features_and_parquet(root: Path) -> None:
    info = load_info(root)
    renamed_features = {}
    for key, ft in info["features"].items():
        renamed_features[FEATURE_RENAME.get(key, key)] = ft
    info["features"] = renamed_features
    write_info(info, root)

    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        stats = {FEATURE_RENAME.get(k, k): v for k, v in stats.items()}
        stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    for pq in tqdm(list((root / "data").glob("*/*.parquet")), desc="rename parquet cols"):
        df = pd.read_parquet(pq)
        df = df.rename(columns={k: v for k, v in FEATURE_RENAME.items() if k in df.columns})
        df.to_parquet(pq, index=False)

    log.info("特征名已对齐 LeRobot LIBERO 约定")


def filter_legacy_metadata_to_local(root: Path, episode_paths: list[Path] | None = None) -> None:
    """仅保留本地已有 parquet 对应的 episodes.jsonl 行（避免 convert 时数量不一致）。"""
    paths = episode_paths if episode_paths is not None else list_episode_parquets(root)
    local_indices = {episode_index_from_path(p) for p in paths}
    ep_path = root / "meta" / "episodes.jsonl"
    if not ep_path.is_file():
        return
    kept = []
    with jsonlines.open(ep_path, "r") as reader:
        for row in reader:
            if row.get("episode_index") in local_indices:
                kept.append(row)
    with jsonlines.open(ep_path, "w") as writer:
        writer.write_all(kept)
    tasks_path = root / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        task_indices = {row.get("task_index") for row in kept if "task_index" in row}
        tasks_kept = []
        with jsonlines.open(tasks_path, "r") as reader:
            for row in reader:
                if row.get("task_index") in task_indices:
                    tasks_kept.append(row)
        with jsonlines.open(tasks_path, "w") as writer:
            writer.write_all(tasks_kept)
    log.info("已过滤 legacy metadata 至 %d 个本地 episode", len(kept))


def sync_info_from_v30_episodes(root: Path) -> None:
    """v3 转换后按 meta/episodes 校正 total_episodes / total_frames。"""
    from lerobot.datasets.utils import load_episodes

    info = load_info(root)
    episodes = load_episodes(root)
    n_eps = len(episodes)
    total_frames = 0
    if "length" in episodes.column_names:
        total_frames = int(sum(episodes["length"]))
    info["total_episodes"] = n_eps
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{n_eps}"}
    write_info(info, root)
    log.info("v3 info 同步: episodes=%d frames=%d", n_eps, total_frames)


def sync_info_episode_counts(root: Path, episode_paths: list[Path] | None = None) -> None:
    """按本地 episode parquet 数量校正 info.json（下载未完成时便于 smoke）。"""
    info = load_info(root)
    if episode_paths is not None:
        ep_paths = episode_paths
    else:
        ep_paths = [
            p for p in (root / "data").glob("*/*.parquet") if "episode_" in p.name
        ]
        ep_paths = sorted(ep_paths, key=episode_index_from_path)
    n_eps = len(ep_paths)
    if n_eps == 0:
        return
    total_frames = 0
    for p in ep_paths:
        total_frames += len(pd.read_parquet(p, columns=["frame_index"]))
    info["total_episodes"] = n_eps
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{n_eps}"}
    write_info(info, root)
    log.info("info.json: total_episodes=%d total_frames=%d", n_eps, total_frames)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="数据集根目录（含 meta/ data/）；默认 data/lerobot/libero_90",
    )
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--validate", action="store_true", help="修复后试加载一条样本")
    p.add_argument("--check-only", action="store_true", help="仅检查是否已是完整 v3.0，不修改数据")
    p.add_argument("--validate-only", action="store_true", help="仅检查完整性并试加载样本")
    p.add_argument("--skip-stats", action="store_true")
    p.add_argument("--skip-convert", action="store_true")
    p.add_argument("--max-episodes", type=int, default=None, help="仅处理前 N 个 episode（smoke）")
    p.add_argument("--allow-partial", action="store_true", help="不强制 3959 集齐也可修复")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = args.root if args.root is not None else dataset_root(args.data_root)
    root.mkdir(parents=True, exist_ok=True)

    if args.check_only:
        ok = is_v3_complete(root)
        print(f"{'OK' if ok else 'INCOMPLETE'} {root}")
        raise SystemExit(0 if ok else 1)

    if args.validate_only:
        if not is_v3_complete(root):
            raise SystemExit(f"数据集未完整: {root}")
        validate_loadable(root)
        print(f"OK validate-only {root}")
        return

    hf_bin = ROOT / ".venv310/bin/hf"
    if not hf_bin.is_file():
        hf_bin = Path("hf")

    if not args.skip_download:
        download_meta(root, hf_bin)
        n_local = count_episode_parquets(root)
        info = load_info(root) if (root / "meta" / "info.json").is_file() else {}
        expected = int(info.get("total_episodes", EXPECTED_EPISODES))
        if n_local < expected and not args.allow_partial:
            resume_data_download(root, hf_bin)
        elif n_local < expected:
            log.warning("本地仅 %d/%d episodes，--allow-partial 继续", n_local, expected)

    if not (root / "meta" / "info.json").is_file():
        raise SystemExit(f"缺少 meta/info.json: {root}")

    n_local = count_episode_parquets(root)
    info = load_info(root)
    expected = int(info.get("total_episodes", EXPECTED_EPISODES))
    if n_local < expected and not args.allow_partial:
        raise SystemExit(f"数据未下完: {n_local}/{expected} episodes，请续传或加 --allow-partial")

    frozen_paths = list_episode_parquets(root)
    if args.max_episodes is not None:
        frozen_paths = frozen_paths[: args.max_episodes]
    log.info("冻结 %d 个本地 episode 用于 stats/convert", len(frozen_paths))

    if not args.skip_stats:
        generate_episodes_stats(root, max_episodes=args.max_episodes, episode_paths=frozen_paths)

    if args.allow_partial or args.max_episodes is not None:
        filter_legacy_metadata_to_local(root, frozen_paths)
        sync_info_episode_counts(root, frozen_paths)

    if not args.skip_convert:
        patch_info_for_conversion(root)
        convert_to_v30(root, episode_paths=frozen_paths)
        rename_features_and_parquet(root)
        ensure_tasks_parquet(root, hf_bin)
        sync_info_from_v30_episodes(root)

    if (root / "meta" / "info.json").is_file():
        final = load_info(root)
        n_data_files = len(list((root / "data").glob("*/*.parquet")))
        print(
            f"OK libero_90 codebase={final.get('codebase_version')} "
            f"episodes={final.get('total_episodes')} data_files={n_data_files}"
        )
        if args.validate or (not args.allow_partial and not args.max_episodes):
            validate_loadable(root)


if __name__ == "__main__":
    main()
