#!/usr/bin/env python3
"""Greedy-decode 10 train + 10 val episodes for manual subtask review.

Reads rows.jsonl produced by train_vlm_mix.py, loads ckpt8/best.pt, and writes
review10/review.json plus the frames those rows already decoded.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import sys

import torch

sys.path.insert(0, "/workspace/openpi/cigai_train/openpi-jiangsuan/src")

import train_vlm_mix as t

OUT = t.ROOT / "review10"
N_EACH = 10
SEED = 0


def pick(groups: dict, split: str, n: int) -> list[tuple[str, int]]:
    by_ds: dict[str, list[int]] = defaultdict(list)
    for sp, ds, ep in groups:
        if sp == split:
            by_ds[ds].append(ep)
    rng = random.Random(SEED)
    for ds in by_ds:
        by_ds[ds] = sorted(set(by_ds[ds]))
        rng.shuffle(by_ds[ds])
    datasets = sorted(by_ds)
    picked: list[tuple[str, int]] = []
    i = 0
    while len(picked) < n and i < n * len(datasets) * 4:
        ds = datasets[i % len(datasets)]
        idx = i // len(datasets)
        if idx < len(by_ds[ds]):
            picked.append((ds, by_ds[ds][idx]))
        i += 1
    return picked


def frame_index(image: str) -> int:
    stem = Path(image).stem
    return int(stem.split("_")[-1])


def main() -> None:
    rows = [json.loads(line) for line in (t.ROOT / "rows.jsonl").read_text().splitlines() if line.strip()]
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["dataset"], int(row["episode"]))].append(row)

    chosen = [("train", ds, ep) for ds, ep in pick(groups, "train", N_EACH)]
    chosen += [("val", ds, ep) for ds, ep in pick(groups, "val", N_EACH)]
    flat: list[dict] = []
    for key in chosen:
        items = sorted(groups[key], key=lambda r: frame_index(r["image"]))
        flat.extend(items)
    print(f"episodes={len(chosen)} frames={len(flat)}", flush=True)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    model = t.build_paligemma()
    state = torch.load(t.CKPT / "best.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"best.pt missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device=device, dtype=dtype)
    model.eval()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(t.TOKENIZER))
    preds: list[str] = []
    ds = t.Rows(flat)
    for start in range(0, len(flat), 4):
        batch = [ds[i] for i in range(start, min(start + 4, len(flat)))]
        preds.extend(t.greedy_texts(model, sp, batch, device, dtype))
        print(f"decoded {len(preds)}/{len(flat)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    frame_dir = OUT / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir()

    cursor = 0
    episodes = []
    for split, dataset, episode in chosen:
        items = sorted(groups[(split, dataset, episode)], key=lambda r: frame_index(r["image"]))
        frames = []
        for row, pred in zip(items, preds[cursor : cursor + len(items)]):
            cursor += 1
            ok = t.norm(pred) == t.norm(row["subtask"])
            name = f"{split}_{dataset}_{int(episode):06d}_{frame_index(row['image']):06d}.jpg"
            shutil.copyfile(row["image"], frame_dir / name)
            frames.append(
                {
                    "frame": frame_index(row["image"]),
                    "label": row["label"],
                    "gold": row["subtask"],
                    "pred": pred,
                    "ok": ok,
                    "image": f"frames/{name}",
                }
            )
        episodes.append(
            {
                "split": split,
                "dataset": dataset,
                "episode": int(episode),
                "task": items[0]["task"],
                "n": len(frames),
                "correct": sum(1 for f in frames if f["ok"]),
                "frames": frames,
            }
        )
    payload = {
        "checkpoint": str(t.CKPT / "best.pt"),
        "seed": SEED,
        "n_train": N_EACH,
        "n_val": N_EACH,
        "episodes": episodes,
    }
    (OUT / "review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    n = sum(ep["n"] for ep in episodes)
    hit = sum(ep["correct"] for ep in episodes)
    print(f"wrote {OUT / 'review.json'} frames={n} exact={hit}/{n}", flush=True)


if __name__ == "__main__":
    main()
