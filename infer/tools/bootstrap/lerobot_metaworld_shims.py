"""MetaWorld / LeRobotDataset 兼容 shim（不修改 src/lerobot，在 train 入口导入）。"""

from __future__ import annotations

import functools
import os
from typing import Any


def apply_metaworld_env_eval_shim() -> None:
    """Eval 时补齐 observation.environment_state（39 维 MT1 obs），与 lerobot/metaworld 数据集一致。"""
    import numpy as np
    from gymnasium import spaces

    from lerobot.envs.metaworld import MetaworldEnv

    if getattr(MetaworldEnv, "_cqtongbot_eval_shim_applied", False):
        return

    _orig_init = MetaworldEnv.__init__
    _orig_format = MetaworldEnv._format_raw_obs

    @functools.wraps(_orig_init)
    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        _orig_init(self, *args, **kwargs)
        if self.obs_type != "pixels_agent_pos":
            return
        spaces_map = dict(self.observation_space.spaces)
        if "environment_state" in spaces_map:
            return
        spaces_map["environment_state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(39,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(spaces_map)

    def _patched_format(self, raw_obs: np.ndarray):
        obs = _orig_format(self, raw_obs)
        if self.obs_type == "pixels_agent_pos" and "environment_state" not in obs:
            obs = dict(obs)
            obs["environment_state"] = np.asarray(raw_obs, dtype=np.float32)
        return obs

    MetaworldEnv.__init__ = _patched_init  # type: ignore[method-assign]
    MetaworldEnv._format_raw_obs = _patched_format  # type: ignore[method-assign]
    MetaworldEnv._cqtongbot_eval_shim_applied = True


def _resolve_hf_dataset_index(dataset, idx: int) -> int:
    mapping = getattr(dataset, "_absolute_to_relative_idx", None)
    if mapping is None:
        return idx
    if idx in mapping:
        return mapping[idx]
    if 0 <= idx < len(dataset.hf_dataset):
        return idx
    raise IndexError(
        f"Frame index {idx} is not in the filtered dataset "
        f"(len={len(dataset.hf_dataset)}). Check episode filter / sampler alignment."
    )


def apply_lerobot_dataset_episode_index_shim() -> None:
    """Episode 子集 + EpisodeAwareSampler 时，将绝对 frame index 映射到 hf_dataset 相对下标。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if getattr(LeRobotDataset, "_cqtongbot_episode_shim_applied", False):
        return

    _orig_getitem = LeRobotDataset.__getitem__

    @functools.wraps(_orig_getitem)
    def _patched_getitem(self, idx: int):
        return _orig_getitem(self, _resolve_hf_dataset_index(self, idx))

    LeRobotDataset.__getitem__ = _patched_getitem  # type: ignore[method-assign]
    LeRobotDataset._cqtongbot_episode_shim_applied = True


def apply_lerobot_dataset_tools_episode_shim() -> None:
    """meta/episodes 单文件布局时仍能加载 episode stats。"""
    from pathlib import Path

    import pandas as pd

    import lerobot.datasets.dataset_tools as dataset_tools
    from lerobot.datasets.lerobot_dataset import DEFAULT_EPISODES_PATH, LeRobotDataset

    if getattr(dataset_tools, "_cqtongbot_episode_meta_shim_applied", False):
        return

    _orig = dataset_tools._load_episode_with_stats

    def _patched_load(src_dataset: LeRobotDataset, episode_idx: int) -> dict:
        try:
            return _orig(src_dataset, episode_idx)
        except (KeyError, FileNotFoundError, IndexError):
            pass

        episodes_dir = src_dataset.root / "meta/episodes"
        parquet_path = None
        for candidate in sorted(episodes_dir.rglob("*.parquet")):
            df = pd.read_parquet(candidate, columns=["episode_index"])
            if episode_idx in df["episode_index"].values:
                parquet_path = candidate
                break
        if parquet_path is None:
            raise FileNotFoundError(f"Episode {episode_idx} not found under {episodes_dir}")

        df = pd.read_parquet(parquet_path)
        return df[df["episode_index"] == episode_idx].iloc[0].to_dict()

    dataset_tools._load_episode_with_stats = _patched_load
    dataset_tools._cqtongbot_episode_meta_shim_applied = True


def apply_metaworld_eval_episode_cap_shim() -> None:
    """矩阵 smoke 可设 METAWORLD_SMOKE_MAX_EPISODE_STEPS 缩短 eval rollout。"""
    cap = int(os.environ.get("METAWORLD_SMOKE_MAX_EPISODE_STEPS", "0") or "0")
    if cap <= 0:
        return

    from lerobot.envs.metaworld import MetaworldEnv

    if getattr(MetaworldEnv, "_cqtongbot_ep_cap_shim_applied", False):
        return

    _orig_make = MetaworldEnv._make_envs_task

    def _patched_make(self, env_name: str):
        env = _orig_make(self, env_name)
        env.max_path_length = min(int(env.max_path_length), cap)
        self._max_episode_steps = env.max_path_length
        return env

    MetaworldEnv._make_envs_task = _patched_make  # type: ignore[method-assign]
    MetaworldEnv._cqtongbot_ep_cap_shim_applied = True


def apply_all_metaworld_shims() -> None:
    apply_metaworld_env_eval_shim()
    apply_metaworld_eval_episode_cap_shim()
    apply_lerobot_dataset_episode_index_shim()
    apply_lerobot_dataset_tools_episode_shim()
