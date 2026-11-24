#!/usr/bin/env python3
"""Probe tongbot 00073: metadata, one frame, wall-x key mapping."""
from __future__ import annotations

from pathlib import Path

import torch

REPO = "/workspace/datasets/00073_20260818_RM_01_yj_21"


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(REPO, root=None)
    feats = list(getattr(meta, "features", {}) or [])
    print(
        f"episodes={meta.total_episodes} frames={meta.total_frames} "
        f"fps={meta.fps} features={feats}"
    )
    tasks = getattr(meta, "tasks", None)
    print("tasks", tasks)
    fps = float(meta.fps)
    delta = {"action": [t / fps for t in range(32)]}
    ds = LeRobotDataset(REPO, root=None, delta_timestamps=delta)
    sample = ds[0]
    keys = sorted(sample.keys())
    print("sample keys", keys)
    for k in keys:
        v = sample[k]
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={getattr(v, 'dtype', type(v))}")
        else:
            s = str(v)
            print(f"  {k}: {s[:160]}")

    action = sample.get("action")
    state = sample.get("observation.state")
    assert action is not None, "missing action"
    assert state is not None, "missing observation.state"
    assert action.shape[-1] == 16, action.shape
    assert state.shape[-1] == 16, state.shape
    task = sample.get("task")
    assert task, "empty task/prompt"
    print("[OK] 00073 sample action", tuple(action.shape), "state", tuple(state.shape))
    print("[OK] task:", task)

    cams = [
        "observation.images.front",
        "observation.images.left",
        "observation.images.right",
    ]
    for cam in cams:
        assert cam in sample, cam
        img = sample[cam]
        print(f"[OK] {cam} {getattr(img, 'shape', type(img))}")

    norm = Path("/workspace/outputs/norm_stats_00073_wallx.json")
    assert norm.is_file(), norm
    print("[OK] norm_stats", norm, "bytes", norm.stat().st_size)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
