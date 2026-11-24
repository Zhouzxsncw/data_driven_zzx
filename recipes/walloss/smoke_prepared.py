#!/usr/bin/env python3
"""One-batch smoke: prepared EE set can feed continuous flow and FAST AR labels.

Does not start training. Exits non-zero if any dataset fails to yield a batch
with a 32-step action chunk and at least one <|action_token_*|> label.
"""

from __future__ import annotations

import sys
from pathlib import Path

CONFIGS = [
    Path("/data/wall-oss/datasets/00036_rm75_ee/sft.yml"),
    Path("/data/wall-oss/datasets/00073_rm75_ee_v30/sft.yml"),
    Path("/data/wall-oss/datasets/00091_rm75_ee_v30/sft.yml"),
    Path("/data/wall-oss/datasets/00108_rm75_ee_v30/sft.yml"),
]


def _smoke_one(path: Path) -> str:
    from wall_x.config.loader import load_config
    from wall_x.data.backends.lerobot.build import load_lerobot_v2

    cfg = load_config(str(path))
    cfg.data.num_workers = 0
    _wrapper, loader, _n = load_lerobot_v2(cfg)
    batch = next(iter(loader))
    action = batch.get("action_chunk")
    if action is None:
        action = batch.get("action")
    if action is None:
        raise RuntimeError(f"batch 没有 action_chunk，键: {sorted(batch)}")
    labels = batch.get("labels")
    input_ids = batch.get("input_ids")
    if labels is None or input_ids is None:
        raise RuntimeError(f"batch 没有 labels/input_ids，键: {sorted(batch)}")
    proc = getattr(_wrapper, "processor", None)
    # Collator keeps the processor on the dataset object, not the wrapper.
    n_supervised = int((labels != -100).sum())
    text = ""
    tokenizer = None
    ds = getattr(loader, "dataset", None)
    collator = getattr(loader, "collate_fn", None)
    tokenizer = getattr(collator, "processor", None)
    if tokenizer is not None and hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    if tokenizer is not None:
        text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
    n_fast = text.count("action_token_")
    if tuple(action.shape[-2:]) != (32, 26) and action.shape[-2] != 32:
        raise RuntimeError(f"action shape {tuple(action.shape)} 不是 horizon 32")
    if n_fast < 1 and n_supervised < 1:
        raise RuntimeError(
            f"没有 FAST 监督。supervised={n_supervised} action_token_={n_fast} shape={tuple(action.shape)}"
        )
    if n_fast < 1:
        raise RuntimeError(
            f"文本里没有 <|action_token_*|>。supervised={n_supervised} shape={tuple(action.shape)}"
        )
    return (
        f"ok action={tuple(action.shape)} supervised_labels={n_supervised} "
        f"action_tokens_in_text={n_fast}"
    )


def main() -> None:
    import os
    import traceback

    import torch.distributed as dist

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29531",
            rank=0,
            world_size=1,
        )
    failed = []
    for path in CONFIGS:
        name = path.parent.name
        print(f"[{name}] start {path}", flush=True)
        try:
            print(f"[{name}] {_smoke_one(path)}", flush=True)
        except Exception as exc:
            failed.append(name)
            print(f"[{name}] FAIL {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    if failed:
        raise SystemExit(f"smoke failed: {failed}")
    print("SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
