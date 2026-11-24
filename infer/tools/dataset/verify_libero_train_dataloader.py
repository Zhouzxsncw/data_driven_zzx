#!/usr/bin/env python3
"""预检：各 LIBERO 套件能否被 LeRobot Dataset 加载（对应 TRAIN 数据管线）。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "train" / "src"))

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
DATA_ROOT = Path(os.environ.get("OFFLINE_DATA_ROOT", ROOT / "data")) / "lerobot"


def resolve_suite(suite: str) -> tuple[Path, str, list[int] | None]:
    suite_root = DATA_ROOT / suite
    combined = DATA_ROOT / "libero"
    idx_file = DATA_ROOT / "_suite_indices" / f"{suite}.txt"
    if (suite_root / "meta/info.json").is_file():
        return suite_root, f"lerobot/{suite}", None
    if (combined / "meta/info.json").is_file() and idx_file.is_file():
        eps = [int(x) for x in idx_file.read_text().split(",") if x.strip()]
        return combined, "lerobot/libero", eps
    return suite_root, f"lerobot/{suite}", None


def main() -> int:
    import lerobot_env_bootstrap

    lerobot_env_bootstrap.bootstrap()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    fail = 0
    print(f"LIBERO TRAIN 数据加载预检 (root={DATA_ROOT})")
    print("=" * 60)
    for suite in SUITES:
        root, repo_id, episodes = resolve_suite(suite)
        if not (root / "meta/info.json").is_file():
            print(f"FAIL {suite}: 无 meta/info.json @ {root}")
            fail += 1
            continue
        try:
            meta = LeRobotDatasetMetadata(repo_id, root=str(root))
            info = json.loads((root / "meta/info.json").read_text())
            codebase = info.get("codebase_version", meta.info.get("codebase_version", "?"))
            ep_note = f"subset={len(episodes)}" if episodes else "full"
            print(
                f"OK   {suite}: repo={repo_id} v={codebase} "
                f"total_episodes={info.get('total_episodes')} ({ep_note})"
            )
            # 可选：试读 1 帧（部分环境 torchvision 无 VideoReader，失败不阻断）
            try:
                kw: dict = {"repo_id": repo_id, "root": str(root), "video_backend": "pyav"}
                if episodes is not None:
                    kw["episodes"] = episodes[:1]
                ds = LeRobotDataset(**kw)
                _ = ds[0]
                print(f"     sample[0] keys={len(ds[0])}")
            except Exception as sample_err:
                print(f"     WARN sample[0]: {type(sample_err).__name__}: {sample_err}")
        except Exception as e:
            print(f"FAIL {suite}: {type(e).__name__}: {e}")
            fail += 1
    print("=" * 60)
    if fail:
        print(f"失败 {fail} 个套件")
        return 1
    print("全部套件 TRAIN 数据加载通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
