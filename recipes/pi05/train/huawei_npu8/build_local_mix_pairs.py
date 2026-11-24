#!/usr/bin/env python3
"""在本机 /home/xk/dataset 把 36、73、91、108 两两拼成 LeRobot 数据集。

视频硬链接。parquet 只改 episode_index / index / task_index，并丢掉 108 的 force6d。
36 和 108 的任务句相同，拼在一起时共用一个 task_index。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/xk/dataset")
CHUNKS = 1000
KEEP = [
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]
PAIRS = [
    ("36", "73"),
    ("36", "91"),
    ("36", "108"),
    ("73", "91"),
    ("73", "108"),
    ("91", "108"),
]
SOURCES = {
    "36": {
        "id": "00036",
        "dir": "00036_20260804_RM_01_yj_v21",
        "keys": {
            "front": "observation.images.front",
            "left": "observation.images.left",
            "right": "observation.images.right",
        },
        "stat_rename": {},
    },
    "73": {
        "id": "00073",
        "dir": "00073_20260818_RM_01_yj_21",
        "keys": {
            "front": "observation.images.front",
            "left": "observation.images.left",
            "right": "observation.images.right",
        },
        "stat_rename": {},
    },
    "91": {
        "id": "00091",
        "dir": "00091_20260903_RM_01_yj_21",
        "keys": {
            "front": "observation.images.front",
            "left": "observation.images.left",
            "right": "observation.images.right",
        },
        "stat_rename": {},
    },
    "108": {
        "id": "00108",
        "dir": "00108_20260922_RM_01_YJ_v21",
        "keys": {
            "front": "observation.images.cam_mid",
            "left": "observation.images.cam_left",
            "right": "observation.images.cam_right",
        },
        "stat_rename": {
            "observation.images.cam_mid": "observation.images.front",
            "observation.images.cam_left": "observation.images.left",
            "observation.images.cam_right": "observation.images.right",
        },
    },
}


def load_task(src: Path) -> str:
    line = (src / "meta" / "tasks.jsonl").read_text().splitlines()[0]
    return json.loads(line)["task"]


def load_lengths(src: Path) -> list[int]:
    lengths = []
    for line in (src / "meta" / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            lengths.append(int(json.loads(line)["length"]))
    return lengths


def load_stats(src: Path) -> list[dict]:
    rows = []
    with (src / "meta" / "episodes_stats.jsonl").open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def scalar_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    qs = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "min": [float(values.min())],
        "max": [float(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.size)],
        "q01": [float(qs[0])],
        "q10": [float(qs[1])],
        "q50": [float(qs[2])],
        "q90": [float(qs[3])],
        "q99": [float(qs[4])],
    }


def adapt_stats(stats: dict, rename: dict, episode: int, task_index: int, frame_offset: int, n: int) -> dict:
    out = {}
    for key, value in stats.items():
        if key.startswith("observation.force6d"):
            continue
        out[rename.get(key, key)] = value
    out["episode_index"] = scalar_stats(np.full(n, episode, dtype=np.float64))
    out["task_index"] = scalar_stats(np.full(n, task_index, dtype=np.float64))
    out["index"] = scalar_stats(np.arange(frame_offset, frame_offset + n, dtype=np.float64))
    missing = [name for name in ("observation.images.front", "observation.images.left", "observation.images.right", "action") if name not in out]
    if missing:
        raise SystemExit(f"stats missing {missing}")
    return out


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.stat().st_ino == src.stat().st_ino and dst.stat().st_dev == src.stat().st_dev:
            return
        dst.unlink()
    os.link(src, dst)


def rewrite(src: Path, dst: Path, episode: int, task_index: int, frame_offset: int) -> int:
    table = pq.read_table(src, columns=KEEP)
    n = table.num_rows
    ts = table.column("timestamp").cast(pa.float32())
    table = table.set_column(table.schema.get_field_index("timestamp"), "timestamp", ts)
    for name, values in (
        ("episode_index", [episode] * n),
        ("task_index", [task_index] * n),
        ("index", list(range(frame_offset, frame_offset + n))),
    ):
        i = table.schema.get_field_index(name)
        arr = pa.array(values, type=pa.int64())
        table = table.set_column(i, name, arr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return n


def build(pair: tuple[str, str], catalog: dict, info_template: dict) -> dict:
    name = "mix_" + "_".join(pair)
    out = ROOT / name
    if out.exists():
        raise SystemExit(f"refuse to overwrite {out}")
    task_strings = []
    task_of = {}
    for key in pair:
        text = catalog[key]["task"]
        if text not in task_of:
            task_of[text] = len(task_strings)
            task_strings.append(text)
        catalog[key]["task_index"] = task_of[text]

    meta = out / "meta"
    meta.mkdir(parents=True)
    episodes = []
    manifest = []
    stats_lines = []
    frame_offset = 0
    episode = 0
    for key in pair:
        src = catalog[key]
        for src_ep, length in enumerate(src["lengths"]):
            chunk = src_ep // CHUNKS
            src_parquet = src["root"] / "data" / f"chunk-{chunk:03d}" / f"episode_{src_ep:06d}.parquet"
            dst_chunk = episode // CHUNKS
            dst_parquet = out / "data" / f"chunk-{dst_chunk:03d}" / f"episode_{episode:06d}.parquet"
            n = rewrite(src_parquet, dst_parquet, episode, src["task_index"], frame_offset)
            if n != length:
                raise SystemExit(f"{name} episode {episode} rows {n} != {length}")
            for slot, src_key in src["keys"].items():
                video_src = src["root"] / "videos" / f"chunk-{chunk:03d}" / src_key / f"episode_{src_ep:06d}.mp4"
                video_dst = out / "videos" / f"chunk-{dst_chunk:03d}" / f"observation.images.{slot}" / f"episode_{episode:06d}.mp4"
                if not video_src.is_file():
                    raise SystemExit(f"missing video {video_src}")
                link(video_src, video_dst)
            stat = adapt_stats(src["stats"][src_ep]["stats"], src["stat_rename"], episode, src["task_index"], frame_offset, n)
            stats_lines.append(json.dumps({"episode_index": episode, "stats": stat}, ensure_ascii=False))
            episodes.append(json.dumps({"episode_index": episode, "tasks": [src["task"]], "length": n}, ensure_ascii=False))
            manifest.append(json.dumps({
                "episode_index": episode,
                "source": src["id"],
                "source_dataset": src["root"].name,
                "source_episode_index": src_ep,
                "task_index": src["task_index"],
                "length": n,
            }, ensure_ascii=False))
            frame_offset += n
            episode += 1
            if episode % 200 == 0:
                print(f"{name} {episode}", flush=True)

    (meta / "episodes.jsonl").write_text("\n".join(episodes) + "\n")
    (meta / "episodes_stats.jsonl").write_text("\n".join(stats_lines) + "\n")
    (meta / "merge_manifest.jsonl").write_text("\n".join(manifest) + "\n")
    (meta / "tasks.jsonl").write_text("".join(
        json.dumps({"task_index": i, "task": text}, ensure_ascii=False) + "\n"
        for i, text in enumerate(task_strings)
    ))
    info = json.loads(json.dumps(info_template))
    info["total_episodes"] = episode
    info["total_frames"] = frame_offset
    info["total_tasks"] = len(task_strings)
    info["total_videos"] = episode * 3
    info["total_chunks"] = (episode + CHUNKS - 1) // CHUNKS
    info["chunks_size"] = CHUNKS
    info["splits"] = {"train": f"0:{episode}"}
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    parts = []
    for key in pair:
        src = catalog[key]
        n_ep = len(src["lengths"])
        n_fr = sum(src["lengths"])
        parts.append({
            "id": src["id"],
            "episodes": n_ep,
            "frames": n_fr,
            "frame_share": n_fr / frame_offset,
            "task_index": src["task_index"],
        })
    summary = {
        "name": name,
        "path": str(out),
        "episodes": episode,
        "frames": frame_offset,
        "tasks": task_strings,
        "same_task_string": len(task_strings) == 1,
        "sources": parts,
    }
    (meta / "combo.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"OK {name} episodes={episode} frames={frame_offset} tasks={len(task_strings)}", flush=True)
    return summary


def main() -> None:
    catalog = {}
    for key, spec in SOURCES.items():
        root = ROOT / spec["dir"]
        lengths = load_lengths(root)
        stats = load_stats(root)
        if len(stats) != len(lengths):
            raise SystemExit(f"{key} stats {len(stats)} != episodes {len(lengths)}")
        catalog[key] = {
            **spec,
            "root": root,
            "task": load_task(root),
            "lengths": lengths,
            "stats": stats,
        }
    info_template = json.loads((ROOT / "mix_108_36_91_73" / "meta" / "info.json").read_text())
    summaries = [build(pair, catalog, info_template) for pair in PAIRS]
    index = ROOT / "mix_pairs_index.json"
    index.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")
    print(f"index {index}", flush=True)


if __name__ == "__main__":
    main()
