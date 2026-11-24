#!/usr/bin/env python3
"""在本机用已经在盘上的四份原料拼 mix_108_36_91_73。

视频硬链接，parquet 改 episode_index / index / task_index。
meta/ 和 norm_stats.json 要先放到输出目录。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/pi05")
OUT = ROOT / "mix_108_36_91_73"
# 盘上的目录名和 manifest 里写的 source_dataset 差一个 v。
ALIASES = {
    "00036_20260804_RM_01_yj_21": "00036_20260804_RM_01_yj_v21",
}
KEYS = {
    "00108_20260922_RM_01_YJ_v21": {
        "front": "observation.images.cam_mid",
        "left": "observation.images.cam_left",
        "right": "observation.images.cam_right",
    },
}
DEFAULT_KEYS = {
    "front": "observation.images.front",
    "left": "observation.images.left",
    "right": "observation.images.right",
}


def src_chunk(dataset: str, episode: int) -> int:
    info = json.loads((ROOT / dataset / "meta" / "info.json").read_text())
    size = int(info.get("chunks_size") or 1000)
    return episode // size


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_ino == src.stat().st_ino and dst.stat().st_dev == src.stat().st_dev:
            return
        dst.unlink()
    os.link(src, dst)


def rewrite(src: Path, dst: Path, episode: int, task_index: int, frame_offset: int) -> int:
    table = pq.read_table(src)
    n = table.num_rows
    schema = table.schema

    def col(name: str, values):
        i = schema.get_field_index(name)
        arr = pa.array(values, type=schema.field(name).type)
        return table.set_column(i, name, arr) if False else arr

    # set_column returns a new table; apply one by one
    out = table
    for name, values in (
        ("episode_index", [episode] * n),
        ("index", list(range(frame_offset, frame_offset + n))),
        ("task_index", [task_index] * n),
    ):
        i = out.schema.get_field_index(name)
        arr = pa.array(values, type=out.schema.field(name).type)
        out = out.set_column(i, name, arr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst)
    return n


def main() -> None:
    manifest_path = OUT / "meta" / "merge_manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"missing {manifest_path}")
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if len(rows) != 1481:
        raise SystemExit(f"manifest rows {len(rows)} != 1481")
    offset = 0
    for n_done, row in enumerate(rows):
        dataset = ALIASES.get(row["source_dataset"], row["source_dataset"])
        src_ep = int(row["source_episode_index"])
        ep = int(row["episode_index"])
        task = int(row["task_index"])
        keys = KEYS.get(row["source_dataset"], DEFAULT_KEYS)
        chunk = src_chunk(dataset, src_ep)
        src_dir = ROOT / dataset
        parquet = src_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{src_ep:06d}.parquet"
        if not parquet.is_file():
            raise SystemExit(f"missing parquet {parquet}")
        dst_chunk = ep // 1000
        dst_parquet = OUT / "data" / f"chunk-{dst_chunk:03d}" / f"episode_{ep:06d}.parquet"
        n = rewrite(parquet, dst_parquet, ep, task, offset)
        if n != int(row["length"]):
            raise SystemExit(f"episode {ep} rows {n} != manifest {row['length']}")
        for name, src_key in keys.items():
            src = src_dir / "videos" / f"chunk-{chunk:03d}" / src_key / f"episode_{src_ep:06d}.mp4"
            dst = OUT / "videos" / f"chunk-{dst_chunk:03d}" / f"observation.images.{name}" / f"episode_{ep:06d}.mp4"
            if not src.is_file():
                raise SystemExit(f"missing video {src}")
            link(src, dst)
        offset += n
        if n_done % 100 == 0:
            print(f"built {n_done}/1481 offset={offset}", flush=True)
    unify_schema(OUT)
    info = json.loads((OUT / "meta" / "info.json").read_text())
    if int(info["total_episodes"]) != 1481 or int(info["total_frames"]) != offset:
        raise SystemExit(f"info {info['total_episodes']} {info['total_frames']} != 1481 {offset}")
    print(f"OK episodes=1481 frames={offset}", flush=True)


def unify_schema(out: Path) -> None:
    """00108 带 force6d 和 double 时间戳，另外三份没有。合成一种 schema。"""
    drop_names = ("observation.force6d.left", "observation.force6d.right")
    rewritten = 0
    for path in sorted((out / "data").rglob("*.parquet")):
        schema = pq.read_schema(path)
        has_force = any(name in schema.names for name in drop_names)
        ts = schema.field("timestamp").type
        if not has_force and ts == pa.float32():
            continue
        table = pq.read_table(path)
        for name in drop_names:
            if name in table.column_names:
                table = table.drop(name)
        col = table.column("timestamp").cast(pa.float32())
        table = table.set_column(table.schema.get_field_index("timestamp"), "timestamp", col)
        pq.write_table(table, path)
        rewritten += 1
    print(f"unified schema, rewrote {rewritten}", flush=True)


if __name__ == "__main__":
    main()
