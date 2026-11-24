#!/usr/bin/env python3
"""PaliGemma CE: image -> the episode's own task sentence.

Sources, half and half inside each batch:
  bridge  /workspace/openpi/datasets/BridgeData2_LeRobot_v3  camera image_0
  mix     /workspace/openpi/datasets/mix_108_36_91_73        camera front

Init is pi05_fixed PaliGemma. Episode split is 10:1 inside each source.
One epoch walks every Bridge training frame once and draws the same number
of COMBINED frames with replacement.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_vlm_mix import greedy_texts, load_paligemma, norm, pack, setup_ddp  # noqa: E402

ROOT = Path("/workspace/openpi/vlm_mix")
CKPT = ROOT / "ckpt_task"
INDEX = ROOT / "task_episodes.jsonl"
BRIDGE = Path("/workspace/openpi/datasets/BridgeData2_LeRobot_v3")
MIX = Path("/workspace/openpi/datasets/mix_108_36_91_73")
BATCH_PER_GPU = 8
HALF = BATCH_PER_GPU // 2
EPOCHS = int(os.environ.get("VLM_EPOCHS", "2"))
PROMPT = "What should the robot do now?\nTask:"
CAP_MAX = 32

av.logging.set_level(av.logging.PANIC)
_CAPS: OrderedDict[str, tuple] = OrderedDict()
DECODE_FAILS = 0


def task_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        parts = [str(x).strip() for x in list(value)]
    except TypeError:
        return str(value).strip()
    return " ".join(p for p in parts if p).strip()


def build_index() -> list[dict]:
    if INDEX.is_file():
        rows = [json.loads(line) for line in INDEX.read_text().splitlines() if line.strip()]
        logging.info("reuse episodes=%s from %s", len(rows), INDEX)
        return rows
    import pandas as pd

    rows: list[dict] = []
    rng_seed = 0
    bridge_files = sorted((BRIDGE / "meta" / "episodes").rglob("*.parquet"))
    if not bridge_files:
        raise SystemExit(f"no bridge episodes under {BRIDGE}")
    cols = [
        "episode_index",
        "tasks",
        "length",
        "videos/observation.images.image_0/chunk_index",
        "videos/observation.images.image_0/file_index",
        "videos/observation.images.image_0/from_timestamp",
    ]
    frame = pd.read_parquet(bridge_files[0], columns=cols)
    bridge_rows = []
    for rec in frame.itertuples(index=False):
        text = task_text(rec.tasks)
        length = int(rec.length)
        file_index = int(rec[4])
        chunk = int(rec[3])
        if length < 1 or not text or file_index < 0 or chunk < 0:
            continue
        video = BRIDGE / (
            f"videos/observation.images.image_0/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        )
        bridge_rows.append(
            {
                "source": "bridge",
                "episode": int(rec.episode_index),
                "length": length,
                "task": text,
                "video": str(video),
                "t0": float(rec[5]),
                "fps": 5.0,
            }
        )
    mix_rows = []
    for line in (MIX / "meta" / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = task_text(rec.get("tasks"))
        length = int(rec["length"])
        ep = int(rec["episode_index"])
        if length < 1 or not text:
            continue
        chunk = ep // 1000
        video = MIX / f"videos/chunk-{chunk:03d}/observation.images.front/episode_{ep:06d}.mp4"
        mix_rows.append(
            {
                "source": "mix",
                "episode": ep,
                "length": length,
                "task": text,
                "video": str(video),
                "t0": 0.0,
                "fps": 15.0,
            }
        )
    import random

    for group in (bridge_rows, mix_rows):
        rng = random.Random(rng_seed)
        ids = list(range(len(group)))
        rng.shuffle(ids)
        n_val = max(1, int(round(len(group) / 11)))
        val_ids = set(ids[:n_val])
        for i, row in enumerate(group):
            row["split"] = "val" if i in val_ids else "train"
        rows.extend(group)
    INDEX.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    logging.info("wrote episodes=%s bridge=%s mix=%s", len(rows), len(bridge_rows), len(mix_rows))
    return rows


def read_rgb(path: str, ts: float) -> Image.Image:
    global DECODE_FAILS
    try:
        hit = _CAPS.get(path)
        if hit is None:
            container = av.open(path)
            stream = container.streams.video[0]
            _CAPS[path] = (container, stream)
            if len(_CAPS) > CAP_MAX:
                _, (old, _) = _CAPS.popitem(last=False)
                old.close()
        else:
            _CAPS.move_to_end(path)
            container, stream = hit
        offset = int(ts / stream.time_base)
        container.seek(offset, stream=stream, backward=True, any_frame=False)
        picked = None
        for fr in container.decode(stream):
            picked = fr
            if fr.time + 1e-3 >= ts:
                break
        if picked is None:
            raise RuntimeError("no frame")
        arr = picked.to_ndarray(format="rgb24")
        return Image.fromarray(arr)
    except Exception:
        DECODE_FAILS += 1
        return Image.new("RGB", (224, 224))


def locate(cum: np.ndarray, episodes: list[dict], fid: int) -> tuple[dict, int]:
    ep_i = int(np.searchsorted(cum, fid, side="right"))
    prev = 0 if ep_i == 0 else int(cum[ep_i - 1])
    return episodes[ep_i], int(fid) - prev


class HalfBatch(Dataset):
    """Each 8-item block is 4 Bridge frames then 4 COMBINED frames."""

    def __init__(
        self,
        bridge_eps: list[dict],
        bridge_cum: np.ndarray,
        mix_eps: list[dict],
        mix_cum: np.ndarray,
        bridge_ids: np.ndarray,
        mix_ids: np.ndarray,
    ):
        if len(bridge_ids) != len(mix_ids):
            raise RuntimeError("bridge/mix sample counts differ")
        if len(bridge_ids) % HALF:
            raise RuntimeError("sample count is not a multiple of 4")
        self.bridge_eps = bridge_eps
        self.bridge_cum = bridge_cum
        self.mix_eps = mix_eps
        self.mix_cum = mix_cum
        self.bridge_ids = bridge_ids
        self.mix_ids = mix_ids

    def __len__(self) -> int:
        return len(self.bridge_ids) * 2

    def __getitem__(self, idx: int) -> dict:
        group = idx // BATCH_PER_GPU
        slot = idx % BATCH_PER_GPU
        if slot < HALF:
            ep, local = locate(self.bridge_cum, self.bridge_eps, int(self.bridge_ids[group * HALF + slot]))
        else:
            ep, local = locate(self.mix_cum, self.mix_eps, int(self.mix_ids[group * HALF + (slot - HALF)]))
        image = read_rgb(ep["video"], float(ep["t0"]) + local / float(ep["fps"]))
        return {"image": image, "prompt": PROMPT, "subtask": ep["task"], "source": ep["source"]}


def cumsum(episodes: list[dict]) -> np.ndarray:
    if not episodes:
        return np.zeros(0, dtype=np.int64)
    return np.cumsum([int(ep["length"]) for ep in episodes], dtype=np.int64)


def epoch_ids(n_frames: int, take: int, epoch: int, rank: int, world: int, salt: int) -> np.ndarray:
    rng = np.random.default_rng(salt + epoch)
    if n_frames < 1 or take < 1:
        raise RuntimeError("empty frame pool")
    if salt == 0:
        perm = rng.permutation(n_frames)
        perm = perm[: world * take]
        return perm[rank::world]
    drawn = rng.integers(0, n_frames, size=world * take, dtype=np.int64)
    return drawn[rank::world]


@torch.no_grad()
def evaluate(model, sp, episodes: list[dict], device, dtype, max_new: int) -> dict | None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    shard = episodes[rank::world]
    model.eval()
    counts = {"bridge": [0, 0], "mix": [0, 0]}
    examples: list[tuple[str, str, str]] = []
    for start in range(0, len(shard), 4):
        batch = []
        for ep in shard[start : start + 4]:
            local = int(ep["length"]) // 2
            image = read_rgb(ep["video"], float(ep["t0"]) + local / float(ep["fps"]))
            batch.append({"image": image, "prompt": PROMPT, "subtask": ep["task"], "source": ep["source"]})
        preds = greedy_texts(model, sp, batch, device, dtype, max_new=max_new)
        for item, pred in zip(batch, preds):
            ok = int(norm(pred) == norm(item["subtask"]))
            slot = counts[item["source"]]
            slot[0] += 1
            slot[1] += ok
            if len(examples) < 4:
                examples.append((item["source"], pred, item["subtask"]))
        if rank == 0 and counts["bridge"][0] + counts["mix"][0] and (counts["bridge"][0] + counts["mix"][0]) % 80 == 0:
            logging.info(
                "val progress rank0 bridge=%s mix=%s",
                counts["bridge"][0],
                counts["mix"][0],
            )
    payload = {"counts": counts, "examples": examples}
    if world > 1:
        gathered: list[dict | None] = [None] * world
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    if rank != 0:
        return None
    merged = {"bridge": [0, 0], "mix": [0, 0]}
    for part in gathered:
        assert part is not None
        for source in merged:
            merged[source][0] += part["counts"][source][0]
            merged[source][1] += part["counts"][source][1]
        for source, pred, gold in part["examples"][:2]:
            logging.info("val source=%s pred=%r gold=%r", source, pred, gold)
    summary = {}
    for source, (n, exact) in merged.items():
        summary[source] = {"n": n, "exact": (exact / n) if n else 0.0}
        logging.info("val source=%s n=%s exact=%.3f", source, n, summary[source]["exact"])
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sys.path.insert(0, "/workspace/openpi/cigai_train/openpi-jiangsuan/src")
    rank, _local, world, device = setup_ddp()
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
    ROOT.mkdir(parents=True, exist_ok=True)
    if rank == 0 and not INDEX.is_file():
        build_index()
    if world > 1:
        dist.barrier()
    episodes = build_index()
    bridge_train = [ep for ep in episodes if ep["source"] == "bridge" and ep["split"] == "train"]
    mix_train = [ep for ep in episodes if ep["source"] == "mix" and ep["split"] == "train"]
    val_eps = [ep for ep in episodes if ep["split"] == "val"]
    bridge_cum = cumsum(bridge_train)
    mix_cum = cumsum(mix_train)
    n_bridge = int(bridge_cum[-1]) if len(bridge_cum) else 0
    n_mix = int(mix_cum[-1]) if len(mix_cum) else 0
    take = (n_bridge // world // HALF) * HALF
    steps_per_epoch = take // HALF
    if steps_per_epoch < 1:
        raise SystemExit(f"not enough bridge frames n={n_bridge} world={world}")
    logging.info(
        "frames bridge=%s mix=%s val_eps=%s steps_per_epoch=%s batch_per_gpu=%s world=%s",
        n_bridge,
        n_mix,
        len(val_eps),
        steps_per_epoch,
        BATCH_PER_GPU,
        world,
    )
    dtype = torch.bfloat16
    model = load_paligemma(device)
    if world > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    n_train = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    logging.info("trainable %.1fM", n_train / 1e6)
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(Path("/workspace/openpi/models/pi05_fixed/tokenizer.model")))
    longest = 1
    for ep in val_eps:
        longest = max(longest, len(sp.encode(ep["task"].strip(), add_bos=False)))
    max_new = min(128, longest + 1)
    logging.info("init weight=pi05_fixed epochs=%s prompt=%r max_new=%s", EPOCHS, PROMPT, max_new)
    opt = torch.optim.AdamW(raw.parameters(), lr=1e-5)
    best = {"bridge": -1.0, "mix": -1.0}
    CKPT.mkdir(exist_ok=True)
    for epoch in range(EPOCHS):
        bridge_ids = epoch_ids(n_bridge, take, epoch, rank, world, salt=0)
        mix_ids = epoch_ids(n_mix, take, epoch, rank, world, salt=10_000)
        ds = HalfBatch(bridge_train, bridge_cum, mix_train, mix_cum, bridge_ids, mix_ids)
        loader = DataLoader(ds, batch_size=BATCH_PER_GPU, shuffle=False, collate_fn=lambda b: b, num_workers=0)
        model.train()
        total = 0.0
        steps = 0
        for batch in loader:
            packed = pack(sp, batch, device, dtype)
            opt.zero_grad(set_to_none=True)
            out = model(
                pixel_values=packed["pixel_values"],
                input_ids=packed["input_ids"],
                attention_mask=packed["attention_mask"],
                labels=packed["labels"],
            )
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
            opt.step()
            steps += 1
            loss_v = loss.detach()
            if world > 1:
                dist.all_reduce(loss_v, op=dist.ReduceOp.SUM)
                loss_v = loss_v / world
            total += float(loss_v)
            if rank == 0 and steps % 10 == 0:
                logging.info(
                    "epoch=%s step=%s loss=%.4f gpus=%s decode_fails=%s",
                    epoch,
                    steps,
                    total / steps,
                    world,
                    DECODE_FAILS,
                )
        if world > 1:
            dist.barrier()
        if rank == 0:
            torch.save(raw.state_dict(), CKPT / "latest.pt")
            logging.info("saved latest before val epoch=%s train_loss=%.4f", epoch, total / max(1, steps))
        metrics = evaluate(raw, sp, val_eps, device, dtype, max_new)
        if rank == 0 and metrics is not None:
            metrics["epoch"] = epoch
            metrics["train_loss"] = total / max(1, steps)
            metrics["steps"] = steps
            (CKPT / f"epoch{epoch}.json").write_text(json.dumps(metrics, indent=2))
            for source, name in (("bridge", "best_bridge.pt"), ("mix", "best_mix.pt")):
                exact = metrics[source]["exact"]
                if exact > best[source]:
                    best[source] = exact
                    torch.save(raw.state_dict(), CKPT / name)
                    logging.info("saved %s exact=%.3f", name, exact)
        if world > 1:
            dist.barrier()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
