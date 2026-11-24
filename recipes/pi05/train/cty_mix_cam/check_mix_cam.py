#!/usr/bin/env python3
"""Fail if a camera-ablation yaml does not mask the dropped slots."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/workspace/openpi/cigai_train/openpi-jiangsuan")
sys.path[:0] = [
    str(REPO / "src"),
    str(REPO / "scripts"),
    str(REPO / "packages/openpi-client/src"),
    str(REPO / "cigia_lerobot/src"),
]

import openpi.training.schedule as _schedule  # noqa: E402
from train_from_yaml import load_train_config  # noqa: E402

SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _sample():
    return {
        "observation.images.front": np.full((8, 8, 3), 10, np.uint8),
        "observation.images.left": np.full((8, 8, 3), 20, np.uint8),
        "observation.images.right": np.full((8, 8, 3), 30, np.uint8),
        "observation.state": np.zeros((16,), np.float32),
        "action": np.zeros((50, 16), np.float32),
    }


def check(yaml: Path) -> None:
    cfg, _framework = load_train_config(yaml)
    dropped = set(str(x) for x in (cfg.data.drop_cameras or ()))
    created = cfg.data.create(cfg.assets_dirs, cfg.model)
    structure = created.repack_transforms.inputs[0].structure
    images = structure["images"]
    missing = [slot for slot in SLOTS if slot in dropped and slot in images]
    present = [slot for slot in SLOTS if slot not in dropped and slot not in images]
    if missing or present:
        raise SystemExit(f"{yaml.name}: repack images={images} dropped={dropped}")
    sample = _sample()
    for transform in created.repack_transforms.inputs:
        sample = transform(sample)
    for transform in created.data_transforms.inputs:
        sample = transform(sample)
    masks = {key: bool(np.asarray(sample["image_mask"][key]).reshape(-1)[0]) for key in SLOTS}
    means = {key: float(np.asarray(sample["image"][key]).mean()) for key in SLOTS}
    for slot in SLOTS:
        if slot in dropped:
            if masks[slot] or means[slot] != 0.0:
                raise SystemExit(f"{yaml.name}: {slot} still live masks={masks} means={means}")
        elif (not masks[slot]) or means[slot] == 0.0:
            raise SystemExit(f"{yaml.name}: {slot} was dropped masks={masks} means={means}")
    print(f"OK {yaml.name} drop={sorted(dropped)} masks={masks} means={means}")


def main() -> None:
    yamls = [Path(arg) for arg in sys.argv[1:]]
    if not yamls:
        raise SystemExit("need yaml paths")
    for yaml in yamls:
        check(yaml)
    cfg, _framework = load_train_config(yamls[0])
    resolved = _schedule.resolve_epoch_schedule(cfg)
    steps = _schedule.steps_per_epoch_from_config(resolved)
    print(
        f"SCHEDULE steps/ep={steps} save_interval={resolved.save_interval} "
        f"num_train_steps={resolved.num_train_steps} "
        f"decay_steps={resolved.lr_schedule.decay_steps}"
    )
    if int(resolved.save_interval) != int(steps):
        raise SystemExit("save_interval is not one epoch")


if __name__ == "__main__":
    main()
