#!/usr/bin/env python3
"""检查 LeRobot 目录能不能被 GR00T UNITREE_G1_SONIC 训练代码吃进去。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_RECIPE = Path(__file__).resolve().parent
if str(_RECIPE) not in sys.path:
    sys.path.insert(0, str(_RECIPE))

from lerobot_sonic_writer import _stats_from_parquets  # noqa: E402
from sonic_schema import ACTION_DIM, STATE_DIM, STATE_SLICES  # noqa: E402

SONIC_STATE = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "projected_gravity",
]
SONIC_ACTION = ["motion_token", "left_hand_joints", "right_hand_joints"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    args = p.parse_args()
    root = args.dataset
    meta = root / "meta"
    errors: list[str] = []
    for name in ("info.json", "modality.json", "episodes.jsonl", "tasks.jsonl"):
        if not (meta / name).is_file():
            errors.append(f"缺 meta/{name}")
    if errors:
        print("\n".join(errors))
        return 1

    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    mod = json.loads((meta / "modality.json").read_text(encoding="utf-8"))
    for k in SONIC_STATE:
        if k not in mod.get("state", {}):
            errors.append(f"modality.json state 缺 {k}")
        else:
            s, e = mod["state"][k]["start"], mod["state"][k]["end"]
            if (s, e) != STATE_SLICES[k]:
                errors.append(f"state.{k} slice {(s, e)} != {STATE_SLICES[k]}")
    for k in SONIC_ACTION:
        if k not in mod.get("action", {}):
            errors.append(f"modality.json action 缺 {k}")
    act = mod.get("action", {})
    if act.get("motion_token", {}).get("end") != 64:
        errors.append("action.motion_token 必须是 64 维")
    if act.get("right_hand_joints", {}).get("end") != ACTION_DIM:
        errors.append(f"action 总维必须是 {ACTION_DIM}")
    if mod.get("video", {}).get("ego_view", {}).get("original_key") != "observation.images.ego_view":
        errors.append("video.ego_view original_key 应对 observation.images.ego_view")
    if "human.task_description" not in mod.get("annotation", {}):
        errors.append("annotation 缺 human.task_description")

    n_ep = int(info.get("total_episodes", 0))
    parquets = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    videos = sorted(root.glob("videos/chunk-*/observation.images.ego_view/episode_*.mp4"))
    if len(parquets) != n_ep:
        errors.append(f"parquet {len(parquets)} != total_episodes {n_ep}")
    if len(videos) != n_ep:
        errors.append(f"mp4 {len(videos)} != total_episodes {n_ep}")
    if n_ep == 0:
        errors.append("数据集是空的")

    if parquets:
        import pandas as pd
        import numpy as np

        df = pd.read_parquet(parquets[0])
        for col in ("observation.state", "action", "annotation.human.task_description"):
            if col not in df.columns:
                errors.append(f"{parquets[0].name} 缺列 {col}")
        if "observation.state" in df.columns:
            st = np.asarray(df["observation.state"].iloc[0], dtype=np.float32)
            if st.shape[-1] != STATE_DIM:
                errors.append(f"observation.state 维 {st.shape} 期望 {STATE_DIM}")
        if "action" in df.columns:
            ac = np.asarray(df["action"].iloc[0], dtype=np.float32)
            if ac.shape[-1] != ACTION_DIM:
                errors.append(f"action 维 {ac.shape} 期望 {ACTION_DIM}（64+7+7）")

    if not (meta / "stats.json").is_file() and parquets:
        stats = _stats_from_parquets(root)
        (meta / "stats.json").write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")
        print("已补写 meta/stats.json")
    elif (meta / "stats.json").is_file():
        stats = json.loads((meta / "stats.json").read_text(encoding="utf-8"))
        for key in ("observation.state", "action"):
            if key not in stats:
                errors.append(f"stats.json 缺 {key}")
            else:
                for field in ("mean", "std", "min", "max", "q01", "q99"):
                    if field not in stats[key]:
                        errors.append(f"stats.json {key} 缺 {field}")

    if errors:
        print("FAIL")
        print("\n".join(errors))
        return 1
    print(
        f"OK  {root}  episodes={n_ep}  frames={info.get('total_frames')}  "
        f"state={STATE_DIM} action={ACTION_DIM}  embodiment=UNITREE_G1_SONIC"
    )
    print("训练: DATASET_PATH=%s bash recipes/gr00t_sonic/train.sh" % root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
