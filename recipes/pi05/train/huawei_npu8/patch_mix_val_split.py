#!/usr/bin/env python3
"""让 mix 训练按 meta/episode_split.json 把训练和验证分开。

验证名单不再逐帧解码视频。train_pytorch 里原来调用的
load_episode_split / split= 并不存在，val_ratio 也没进 DataConfig。
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

ROOT = Path("/home/pi05/openpi_src")
STAMP = time.strftime("%Y%m%d%H%M%S")


def backup(path: Path) -> None:
    dst = path.with_suffix(path.suffix + f".bak-{STAMP}")
    if not dst.exists():
        shutil.copy2(path, dst)


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected 1 occurrence, found {n} for:\n{old[:180]}")
    path.write_text(text.replace(old, new, 1))


def main() -> None:
    config = ROOT / "src/openpi/training/config.py"
    loader = ROOT / "src/openpi/training/data_loader.py"
    train = ROOT / "scripts/train_pytorch.py"
    schedule = ROOT / "src/openpi/training/schedule.py"
    for path in (config, loader, train, schedule):
        backup(path)

    replace_once(
        config,
        """    subtask_repeat: dict[str, float] | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
""",
        """    subtask_repeat: dict[str, float] | None = None
    # YAML: data.val_ratio / data.val_seed. 0 means train on every episode.
    val_ratio: float = 0.0
    val_seed: int = 42

    # Only used for RLDS data loader (ie currently only used for DROID).
""",
    )
    replace_once(
        config,
        """    # Source-frame step. YAML: data.frame_stride: 3 → native 15 Hz as 5 Hz.
    frame_stride: int = 1
""",
        """    # Source-frame step. YAML: data.frame_stride: 3 → native 15 Hz as 5 Hz.
    frame_stride: int = 1
    # YAML: data.val_ratio: 0.1  Copied onto DataConfig so the train loop can see it.
    val_ratio: float = 0.0
    val_seed: int = 42
""",
    )
    replace_once(
        config,
        """            frame_stride=int(self.frame_stride),
            subtask_repeat=dict(self.subtask_repeat) if self.subtask_repeat else None,
        )
""",
        """            frame_stride=int(self.frame_stride),
            subtask_repeat=dict(self.subtask_repeat) if self.subtask_repeat else None,
            val_ratio=float(self.val_ratio),
            val_seed=int(self.val_seed),
        )
""",
    )

    replace_once(loader, "import logging\n", "import json\nimport logging\n")
    replace_once(
        loader,
        '''class FilterByEpisode:
    """Keep only samples whose episode_index is in ``allow`` (used for offline val)."""

    def __init__(self, dataset, allow: set[int]):
        self._dataset = dataset
        self._idxs: list[int] = []
        for i in range(len(dataset)):
            sample = dataset[i]
            ep = sample.get("episode_index", sample.get("episode"))
            if ep is None:
                continue
            ep_i = int(ep.item()) if hasattr(ep, "item") else int(ep)
            if ep_i in allow:
                self._idxs.append(i)
        if not self._idxs:
            raise ValueError(f"FilterByEpisode: 0 frames in allowlist size={len(allow)}")

    def __getitem__(self, index):
        return self._dataset[self._idxs[index]]

    def __len__(self) -> int:
        return len(self._idxs)
''',
        '''def load_episode_split(repo_id: str, val_ratio: float, val_seed: int) -> tuple[set[int], set[int]]:
    """Read meta/episode_split.json. val_ratio and val_seed must match the file."""
    path = pathlib.Path(repo_id) / "meta" / "episode_split.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing episode split: {path}")
    payload = json.loads(path.read_text())
    file_ratio = float(payload.get("val_ratio", val_ratio))
    file_seed = int(payload.get("seed", val_seed))
    if abs(file_ratio - float(val_ratio)) > 1e-6 or file_seed != int(val_seed):
        raise ValueError(
            f"{path} is val_ratio={file_ratio} seed={file_seed}, yaml asked {val_ratio} / {val_seed}"
        )
    train = {int(x) for x in payload["train_episodes"]}
    val = {int(x) for x in payload["val_episodes"]}
    if train & val:
        raise ValueError(f"{path} train/val overlap")
    return train, val


def _lerobot_base(dataset):
    seen: set[int] = set()
    cur = dataset
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "episode_data_index"):
            return cur
        cur = getattr(cur, "_dataset", None)
    return None


class FilterByEpisode:
    """Keep frames whose episode is in ``allow``. Uses episode_data_index, not video decode."""

    def __init__(self, dataset, allow: set[int]):
        self._dataset = dataset
        base = _lerobot_base(dataset)
        if base is None:
            raise ValueError("FilterByEpisode: dataset has no episode_data_index")
        if len(dataset) != len(base):
            raise ValueError(
                f"FilterByEpisode: wrapper length {len(dataset)} != base {len(base)}; "
                "frame_stride or subtask_repeat is on, and this split does not handle that"
            )
        edi = base.episode_data_index
        froms = edi["from"]
        tos = edi["to"]
        allow = {int(x) for x in allow}
        n = len(base)
        idxs: list[int] = []
        for ep in range(len(froms)):
            if ep not in allow:
                continue
            start = int(froms[ep])
            end = int(tos[ep])
            if start < 0 or end > n or end < start:
                raise ValueError(f"episode {ep} frame range {start}:{end} outside 0:{n}")
            idxs.extend(range(start, end))
        if not idxs:
            raise ValueError(f"FilterByEpisode: 0 frames in allowlist size={len(allow)}")
        self._idxs = idxs

    def __getitem__(self, index):
        return self._dataset[self._idxs[index]]

    def __len__(self) -> int:
        return len(self._idxs)
''',
    )

    replace_once(
        train,
        '''def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()
''',
        '''def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework.
    # When val_ratio > 0, train only on the train episode list.
    allow = None
    factory = config.data
    val_ratio = float(getattr(factory, "val_ratio", 0.0) or 0.0)
    repo_id = getattr(factory, "repo_id", None)
    if val_ratio > 0 and repo_id:
        train_ids, _val_ids = _data.load_episode_split(
            repo_id, val_ratio, int(getattr(factory, "val_seed", 42))
        )
        allow = train_ids
    data_loader = _data.create_data_loader(
        config, framework="pytorch", shuffle=True, episode_allowlist=allow
    )
    return data_loader, data_loader.data_config()
''',
    )
    replace_once(
        train,
        '''    val_loader = None
    val_ratio = float(getattr(data_config, "val_ratio", 0.0) or 0.0)
    if val_ratio > 0:
        val_seed = int(getattr(data_config, "val_seed", 42))
        train_ids, val_ids = _data.load_episode_split(data_config.repo_id, val_ratio, val_seed)
        n_val_batches = _data.epoch_batches_for_split(config, "val", world_size)
        if is_main:
            logging.info(
                "VALID split val_ratio=%s seed=%s train_episodes=%s val_episodes=%s (%.2f:1) "
                "eval_every_epoch batches/rank=%s",
                val_ratio,
                val_seed,
                len(train_ids),
                len(val_ids),
                len(train_ids) / len(val_ids),
                n_val_batches,
            )
        val_loader = _data.create_data_loader(
            config,
            framework="pytorch",
            shuffle=False,
            num_batches=n_val_batches,
            split="val",
            num_workers=2,
        )
''',
        '''    val_loader = None
    val_ratio = float(getattr(data_config, "val_ratio", 0.0) or 0.0)
    if val_ratio > 0:
        val_seed = int(getattr(data_config, "val_seed", 42))
        train_ids, val_ids = _data.load_episode_split(data_config.repo_id, val_ratio, val_seed)
        n_val_frames = _data.count_frames_for_episodes(config, val_ids)
        n_val_batches = n_val_frames // int(config.batch_size)
        if n_val_batches < 1:
            raise ValueError(f"val split has {n_val_frames} frames, fewer than one global batch")
        if is_main:
            logging.info(
                "VALID split val_ratio=%s seed=%s train_episodes=%s val_episodes=%s "
                "val_frames=%s batches/rank=%s",
                val_ratio,
                val_seed,
                len(train_ids),
                len(val_ids),
                n_val_frames,
                n_val_batches,
            )
        val_loader = _data.create_data_loader(
            config,
            framework="pytorch",
            shuffle=False,
            num_batches=n_val_batches,
            episode_allowlist=val_ids,
        )
''',
    )
    # count_frames_for_episodes lives next to create_data_loader so it uses the same dataset.
    replace_once(
        loader,
        '''def create_data_loader(
''',
        '''def count_frames_for_episodes(config: _config.TrainConfig, episodes: set[int]) -> int:
    """Frame count of one split, using the same dataset construction as training."""
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    return len(FilterByEpisode(dataset, episodes))


def create_data_loader(
''',
    )
    replace_once(
        schedule,
        '''    dataset = _data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    return len(dataset), False
''',
        '''    dataset = _data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    val_ratio = float(getattr(data_config, "val_ratio", 0.0) or 0.0)
    if val_ratio > 0 and data_config.repo_id:
        train_ids, _val_ids = _data_loader.load_episode_split(
            data_config.repo_id, val_ratio, int(getattr(data_config, "val_seed", 42))
        )
        dataset = _data_loader.FilterByEpisode(dataset, train_ids)
        logging.info(
            "epoch length uses train split only: %s frames, %s episodes",
            len(dataset),
            len(train_ids),
        )
    return len(dataset), False
''',
    )
    print("patched", STAMP)


if __name__ == "__main__":
    main()
