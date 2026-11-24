#!/usr/bin/env python3
"""校验并修复本地 lerobot/metaworld 数据集（不改 src/lerobot）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# push-back-v3 (task_id=37) 与 push-v3 (task_id=38) 在原始数据中共用 task_index=40
PUSH_BACK_TASK_ID = 37
PUSH_V3_TASK_ID = 38
PUSH_BACK_NEW_TASK_INDEX = 49
PUSH_V3_TASK_INDEX = 40


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_config() -> dict:
    cfg_path = _repo_root() / "src/lerobot/envs/metaworld_config.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _episode_task_ids(metaworld_root: Path) -> dict[int, int]:
    import pandas as pd

    ep_to_tid: dict[int, int] = {}
    for parquet_path in sorted((metaworld_root / "data").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index", "task_id"])
        for ep_idx, grp in df.groupby("episode_index"):
            if int(ep_idx) not in ep_to_tid:
                ep_to_tid[int(ep_idx)] = int(grp["task_id"].iloc[0])
    return ep_to_tid


def _push_collision_already_fixed(metaworld_root: Path) -> bool:
    import pandas as pd

    sample = next((metaworld_root / "data").rglob("*.parquet"), None)
    if sample is None:
        return False
    df = pd.read_parquet(sample, columns=["task_id", "task_index"])
    back = df.loc[df["task_id"] == PUSH_BACK_TASK_ID, "task_index"].unique()
    return len(back) == 1 and int(back[0]) == PUSH_BACK_NEW_TASK_INDEX


def fix_push_task_index_collision(metaworld_root: Path, dry_run: bool = False) -> list[str]:
    """将 push-back-v3 的 task_index 从 40 拆到 49，并同步 tasks.parquet。"""
    import pandas as pd

    if _push_collision_already_fixed(metaworld_root):
        return ["push task_index collision already fixed (skip)"]

    cfg = _load_config()
    logs: list[str] = []
    changed_files = 0
    changed_rows = 0

    for parquet_path in sorted((metaworld_root / "data").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        if "task_id" not in df.columns or "task_index" not in df.columns:
            continue
        mask = df["task_id"] == PUSH_BACK_TASK_ID
        if not mask.any():
            continue
        n = int(mask.sum())
        if dry_run:
            logs.append(f"would patch {parquet_path.name}: {n} rows task_id=37 -> task_index={PUSH_BACK_NEW_TASK_INDEX}")
            changed_files += 1
            changed_rows += n
            continue
        df.loc[mask, "task_index"] = PUSH_BACK_NEW_TASK_INDEX
        df.to_parquet(parquet_path, index=False)
        logs.append(f"patched {parquet_path}: {n} rows")
        changed_files += 1
        changed_rows += n

    tasks_path = metaworld_root / "meta/tasks.parquet"
    tasks_df = pd.read_parquet(tasks_path)

    push_v3_desc = cfg["TASK_DESCRIPTIONS"]["push-v3"]
    push_back_desc = "Push the puck back to a goal"

    if dry_run:
        logs.append(f"would rewrite {tasks_path}")
        return logs

    # 重建 tasks：保留非 push 行，写入两条 push 描述
    rows = []
    for desc, row in tasks_df.iterrows():
        ti = int(row["task_index"])
        if ti == PUSH_V3_TASK_INDEX and "push" in str(desc).lower() and "puck" in str(desc).lower():
            continue
        rows.append({"description": str(desc), "task_index": ti})

    rows.append({"description": push_v3_desc, "task_index": PUSH_V3_TASK_INDEX})
    rows.append({"description": push_back_desc, "task_index": PUSH_BACK_NEW_TASK_INDEX})

    new_df = pd.DataFrame(rows).drop_duplicates(subset=["task_index"], keep="last")
    new_df = new_df.set_index("description")
    new_df.index.name = None
    new_df = new_df.sort_values("task_index")
    new_df.to_parquet(tasks_path)

    info_path = metaworld_root / "meta/info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["total_tasks"] = len(new_df)
        info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    logs.append(
        f"tasks.parquet: {len(tasks_df)} -> {len(new_df)} rows; "
        f"data files touched={changed_files} rows={changed_rows}"
    )
    return logs


def verify_dataset(metaworld_root: Path) -> int:
    import pandas as pd

    cfg = _load_config()
    info_path = metaworld_root / "meta/info.json"
    if not info_path.is_file():
        print(f"FAIL: missing {info_path}", file=sys.stderr)
        return 1

    info = json.loads(info_path.read_text(encoding="utf-8"))
    ep_to_tid = _episode_task_ids(metaworld_root)

    errors = []
    for task_key, task_id in sorted(cfg["TASK_NAME_TO_ID"].items(), key=lambda x: x[1]):
        count = sum(1 for tid in ep_to_tid.values() if tid == task_id)
        if count != 50:
            errors.append(f"{task_key} (task_id={task_id}): expected 50 episodes, got {count}")

    tasks_df = pd.read_parquet(metaworld_root / "meta/tasks.parquet")
    if len(tasks_df) < 50:
        errors.append(f"tasks.parquet has {len(tasks_df)} rows (expect >= 50 unique descriptions)")

    # task_index 在 push 修复后应能区分 37/38
    tid_to_tidx: dict[int, set[int]] = {}
    for pq in (metaworld_root / "data").rglob("*.parquet"):
        df = pd.read_parquet(pq, columns=["task_id", "task_index"])
        for tid in df["task_id"].unique():
            tid = int(tid)
            tid_to_tidx.setdefault(tid, set()).update(int(x) for x in df.loc[df["task_id"] == tid, "task_index"].unique())
    if len(tid_to_tidx.get(PUSH_BACK_TASK_ID, set())) > 1:
        errors.append(f"push-back-v3 task_id={PUSH_BACK_TASK_ID} has multiple task_index: {tid_to_tidx[PUSH_BACK_TASK_ID]}")
    if len(tid_to_tidx.get(PUSH_V3_TASK_ID, set())) > 1:
        errors.append(f"push-v3 task_id={PUSH_V3_TASK_ID} has multiple task_index: {tid_to_tidx[PUSH_V3_TASK_ID]}")

    print(f"OK: root={metaworld_root}")
    print(f"     total_episodes={info.get('total_episodes')} tasks_in_parquet={len(tasks_df)}")
    print(f"     episode task_ids: {len(set(ep_to_tid.values()))} unique / {len(ep_to_tid)} episodes")

    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metaworld-root",
        type=Path,
        default=None,
        help="默认 <repo>/data/lerobot/metaworld",
    )
    parser.add_argument(
        "--fix-push-collision",
        action="store_true",
        help="修复 push-v3 / push-back-v3 共用 task_index=40",
    )
    parser.add_argument("--regenerate-indices", action="store_true", help="重写 _metaworld_indices/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    metaworld_root = (args.metaworld_root or root / "data/lerobot/metaworld").resolve()
    if not metaworld_root.is_dir() and (root / "data/lerobot/metaworld_mt50").is_dir():
        metaworld_root = (root / "data/lerobot/metaworld_mt50").resolve()

    if args.fix_push_collision:
        for line in fix_push_task_index_collision(metaworld_root, dry_run=args.dry_run):
            print(line)

    rc = verify_dataset(metaworld_root)
    if rc != 0 and not args.fix_push_collision:
        return rc

    if args.fix_push_collision and not args.dry_run:
        rc = verify_dataset(metaworld_root)
        if rc != 0:
            return rc

    if args.regenerate_indices and not args.dry_run:
        import subprocess

        subprocess.check_call(
            [
                sys.executable,
                str(root / "tools/dataset/generate_metaworld_task_episode_indices.py"),
                "--metaworld-root",
                str(metaworld_root),
            ]
        )
        print("Regenerated _metaworld_indices/")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
