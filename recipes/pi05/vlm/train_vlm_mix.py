#!/usr/bin/env python3
"""Mix 00036/00073/00091/00108 PaliGemma CE on cty-18.

Labels come from the gripper FSM already used by annotate (not human text).
Episode split is 10:1 inside each dataset. Vision is trained together with
the language stack; the earlier 00036 run froze SigLIP and collapsed.

Init: pi0.5 step 016815 paligemma tensors. Action expert is not updated.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

ROOT = Path("/workspace/openpi/vlm_mix")
CKPT = ROOT / "ckpt_atomic"
BATCH_PER_GPU = 8
EPOCHS = int(os.environ.get("VLM_EPOCHS", "10"))
DATA = Path("/workspace/openpi/datasets")
# π0.5 预训练基座。不加载 TongBot 流匹配之后的 PaliGemma。
WEIGHT = Path(os.environ.get("VLM_WEIGHT", "/workspace/openpi/models/pi05_fixed"))
ATOMIC = {
    "approach": "move",
    "transit": "move",
    "open_drawer": "move",
    "close_drawer": "move",
    "pick_banana_or_lemon": "pick",
    "pick_vegetables": "pick",
    "pick_from_belt": "pick",
    "grasp_box": "pick",
    "place_in_black": "place",
    "place_in_blue": "place",
    "place_left": "place",
    "place_right": "place",
    "place_in_green": "place",
}
TOKENIZER = Path("/workspace/openpi/models/pi05_fixed/tokenizer.model")
IMAGE_TOKEN_ID = 257152
N_IMAGE_TOKENS = 256
PROMPT = "Task: {task}\nWhat is the current subtask?\nSubtask:"

DATASETS = [
    {
        "id": "00036",
        "name": "00036_20260804_RM_01_yj_v21",
        "camera": "observation.images.front",
        "fsm": "fruit_veg",
        "open_thr": 0.70,
        "close_thr": 0.40,
        "task": (
            "With both hands, pick up the yellow banana and the yellow lemon and put them "
            "into the black box; then take the green vegetables and place them into the blue box."
        ),
        "templates": {
            "approach": "move",
            "pick_banana_or_lemon": "pick",
            "place_in_black": "place",
            "pick_vegetables": "pick",
            "place_in_blue": "place",
            "transit": "move",
        },
    },
    {
        "id": "00073",
        "name": "00073_20260818_RM_01_yj_21",
        "camera": "observation.images.front",
        "fsm": "conveyor",
        "open_thr": 0.80,
        "close_thr": 0.50,
        "task": (
            "Using both hands, pick up the courier items from the green conveyor belt. "
            "Place the white courier bags into the left box and the brown courier boxes into the right box."
        ),
        "templates": {
            "approach": "move",
            "pick_from_belt": "pick",
            "place_left": "place",
            "place_right": "place",
            "transit": "move",
        },
    },
    {
        "id": "00091",
        "name": "00091_20260903_RM_01_yj_21",
        "camera": "observation.images.front",
        "fsm": "drawer_medbox",
        "open_thr": 0.70,
        "close_thr": 0.40,
        "task": (
            "Using both hands, pull open the drawer with the blue handle using your left hand, "
            "take out the white medicine box with your right hand, place it into the green drawer, "
            "and finally close the drawer."
        ),
        "templates": {
            "approach": "move",
            "open_drawer": "move",
            "grasp_box": "pick",
            "place_in_green": "place",
            "close_drawer": "move",
            "transit": "move",
        },
    },
    {
        "id": "00108",
        "name": "00108_20260922_RM_01_YJ_v21",
        "camera": "observation.images.cam_mid",
        "fsm": "fruit_veg",
        "open_thr": 0.70,
        "close_thr": 0.40,
        "task": (
            "With both hands, pick up the yellow banana and the yellow lemon and put them "
            "into the black box; then take the green vegetables and place them into the blue box."
        ),
        "templates": {
            "approach": "move",
            "pick_banana_or_lemon": "pick",
            "place_in_black": "place",
            "pick_vegetables": "pick",
            "place_in_blue": "place",
            "transit": "move",
        },
    },
]


@dataclass
class Ev:
    frame: int
    hand: str
    kind: str


def hysteresis(x: np.ndarray, open_thr: float, close_thr: float) -> np.ndarray:
    hi, lo = float(open_thr), float(close_thr)
    if hi < lo:
        hi, lo = lo, hi
    out = np.empty(len(x), np.int8)
    x0 = float(x[0]) if len(x) else 0.0
    if x0 >= hi:
        state = 0
    elif x0 <= lo:
        state = 1
    else:
        state = 0 if x0 >= 0.5 * (hi + lo) else 1
    for i, v in enumerate(x):
        v = float(v)
        if v >= hi:
            state = 0
        elif v <= lo:
            state = 1
        out[i] = state
    return out


def edges(x: np.ndarray, hand: str, open_thr: float, close_thr: float, debounce: int) -> list[Ev]:
    binary = hysteresis(x, open_thr, close_thr)
    events: list[Ev] = []
    run_val = int(binary[0]) if len(binary) else 0
    run_len = 0
    stable = run_val
    for i, v in enumerate(binary.tolist()):
        v = int(v)
        if v == run_val:
            run_len += 1
        else:
            run_val = v
            run_len = 1
        if run_len >= debounce and v != stable:
            events.append(Ev(max(0, i - debounce + 1), hand, "close" if v == 1 else "open"))
            stable = v
    return events


def fruit_labels(events: list[Ev]) -> dict[tuple[int, str, str], str]:
    pending: dict[str, Ev] = {}
    cycles: list[tuple[Ev, Ev | None]] = []
    stray: list[Ev] = []
    for ev in events:
        if ev.kind == "close":
            prev = pending.pop(ev.hand, None)
            if prev is not None:
                cycles.append((prev, None))
            pending[ev.hand] = ev
        else:
            close = pending.pop(ev.hand, None)
            if close is not None:
                cycles.append((close, ev))
            else:
                stray.append(ev)
    cycles.extend((c, None) for c in pending.values())
    cycles.sort(key=lambda p: p[0].frame)
    n_fruit = len(cycles) if len(cycles) <= 1 else min(2, len(cycles) - 1)
    labels: dict[tuple[int, str, str], str] = {}
    for i, (close, open_ev) in enumerate(cycles):
        fruit = i < n_fruit
        labels[(close.frame, close.hand, "close")] = "pick_banana_or_lemon" if fruit else "pick_vegetables"
        if open_ev is not None:
            labels[(open_ev.frame, open_ev.hand, "open")] = "place_in_black" if fruit else "place_in_blue"
    n_black = sum(1 for v in labels.values() if v == "place_in_black")
    for ev in stray:
        fruit = n_black < n_fruit
        if fruit:
            n_black += 1
        labels[(ev.frame, ev.hand, ev.kind)] = "place_in_black" if fruit else "place_in_blue"
    return labels


def segments(ep: int, n: int, events: list[Ev], fsm: str, min_len: int = 8) -> list[tuple[int, int, str]]:
    label = "approach"
    t0 = 0
    out: list[tuple[int, int, str]] = []
    fmap = fruit_labels(events) if fsm == "fruit_veg" else None

    def emit(a: int, b: int, lab: str) -> None:
        if b - a >= min_len:
            out.append((a, b, lab))

    for ev in events:
        if fsm == "fruit_veg":
            new = fmap.get((ev.frame, ev.hand, ev.kind), "transit")
        elif fsm == "conveyor":
            new = "pick_from_belt" if ev.kind == "close" else ("place_left" if ev.hand == "left" else "place_right")
        elif fsm == "drawer_medbox":
            new = "transit"
        else:
            new = "transit"
        if fsm == "drawer_medbox":
            pass
        if ev.frame > t0:
            emit(t0, ev.frame, label)
            t0 = ev.frame
            label = new
    emit(t0, n, label)
    if fsm != "drawer_medbox":
        return out
    # Re-run drawer with the annotate state machine.
    segs: list[tuple[int, int, str]] = []
    left_close = right_close = 0
    label = "approach"
    t0 = 0
    for ev in events:
        new = "transit"
        if ev.hand == "left" and ev.kind == "close":
            left_close += 1
            new = "open_drawer" if left_close == 1 else "close_drawer"
        elif ev.hand == "right" and ev.kind == "close":
            right_close += 1
            new = "grasp_box"
        elif ev.kind == "open" and ev.hand == "right" and right_close >= 1:
            new = "place_in_green"
        elif ev.kind == "open" and ev.hand == "left" and right_close >= 1:
            new = "place_in_green"
        elif ev.kind == "open" and ev.hand == "left" and left_close == 0:
            new = "approach"
        if ev.frame > t0 and ev.frame - t0 >= min_len:
            segs.append((t0, ev.frame, label))
        if ev.frame > t0:
            t0 = ev.frame
            label = new
    if n - t0 >= min_len:
        segs.append((t0, n, label))
    return segs


def episode_action(ds: Path, ep: int) -> np.ndarray:
    chunk = ep // 1000
    path = ds / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
    table = pq.read_table(path, columns=["action"])
    col = table.column("action").to_pylist()
    return np.asarray(col, dtype=np.float32)


def grab_frames(video: Path, indexes: dict[int, Path]) -> None:
    """One decode pass. indexes maps frame index -> jpg path."""
    if not indexes:
        return
    needed = {i for i, path in indexes.items() if not path.is_file()}
    if not needed:
        return
    container = av.open(str(video))
    stream = container.streams.video[0]
    last = max(needed)
    saved: set[int] = set()
    for i, fr in enumerate(container.decode(stream)):
        if i in needed:
            img = fr.to_image().convert("RGB").resize((224, 224), Image.BICUBIC)
            indexes[i].parent.mkdir(parents=True, exist_ok=True)
            img.save(indexes[i], quality=90)
            saved.add(i)
        if i >= last and saved == needed:
            break
    container.close()
    if saved != needed:
        raise RuntimeError(f"{video} missing frames {sorted(needed - saved)[:4]}")


def apply_atomic(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        label = row["label"]
        if label not in ATOMIC:
            raise KeyError(label)
        copied = dict(row)
        copied["subtask"] = ATOMIC[label]
        out.append(copied)
    return out


def build_rows() -> list[dict]:
    rows_path = ROOT / "rows_atomic.jsonl"
    if rows_path.is_file():
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        logging.info("reuse %s atomic rows from %s", len(rows), rows_path)
        return rows
    src = ROOT / "rows.jsonl"
    if src.is_file():
        rows = apply_atomic(
            [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
        )
        rows_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        logging.info("remapped %s rows to pick/move/place -> %s", len(rows), rows_path)
        return rows
    rng = random.Random(0)
    rows: list[dict] = []
    for spec in DATASETS:
        ds = DATA / spec["name"]
        info = json.loads((ds / "meta/info.json").read_text())
        n_ep = int(info["total_episodes"])
        ids = list(range(n_ep))
        rng.shuffle(ids)
        n_val = max(1, int(round(n_ep / 11)))
        val_ids = set(ids[:n_val])
        logging.info("%s episodes=%s train=%s val=%s", spec["id"], n_ep, n_ep - n_val, n_val)
        for ep in range(n_ep):
            action = episode_action(ds, ep)
            evs: list[Ev] = []
            if action.ndim == 2 and action.shape[1] > 15:
                evs += edges(action[:, 14], "left", spec["open_thr"], spec["close_thr"], 8)
                evs += edges(action[:, 15], "right", spec["open_thr"], spec["close_thr"], 8)
                evs.sort(key=lambda e: (e.frame, e.hand))
            segs = segments(ep, len(action), evs, spec["fsm"])
            split = "val" if ep in val_ids else "train"
            chunk = ep // 1000
            video = ds / f"videos/chunk-{chunk:03d}/{spec['camera']}/episode_{ep:06d}.mp4"
            pending: dict[int, Path] = {}
            staged: list[tuple[Path, str, str]] = []
            for t0, t1, label in segs:
                text = spec["templates"].get(label)
                if not text:
                    continue
                mid = (t0 + t1) // 2
                jpg = ROOT / "frames" / spec["id"] / f"{ep:06d}_{mid:06d}.jpg"
                pending[mid] = jpg
                staged.append((jpg, label, text))
            grab_frames(video, pending)
            for jpg, label, text in staged:
                rows.append(
                    {
                        "dataset": spec["id"],
                        "episode": ep,
                        "split": split,
                        "label": label,
                        "task": spec["task"],
                        "subtask": ATOMIC[label],
                        "image": str(jpg),
                    }
                )
            if ep % 25 == 0:
                logging.info("%s ep %s/%s rows=%s", spec["id"], ep, n_ep, len(rows))
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = {}
    for r in rows:
        counts.setdefault(r["dataset"] + ":" + r["split"], 0)
        counts[r["dataset"] + ":" + r["split"]] += 1
    logging.info("rows %s", counts)
    return rows


def build_paligemma():
    from transformers import PaliGemmaForConditionalGeneration
    from transformers.models.auto import CONFIG_MAPPING

    import openpi.models.gemma as gemma

    vlm = gemma.get_config("gemma_2b")
    cfg = CONFIG_MAPPING["paligemma"]()
    cfg._vocab_size = 257152  # noqa: SLF001
    cfg.image_token_index = 257152
    cfg.text_config.hidden_size = vlm.width
    cfg.text_config.intermediate_size = vlm.mlp_dim
    cfg.text_config.num_attention_heads = vlm.num_heads
    cfg.text_config.head_dim = vlm.head_dim
    cfg.text_config.num_hidden_layers = vlm.depth
    cfg.text_config.num_key_value_heads = vlm.num_kv_heads
    cfg.text_config.hidden_activation = "gelu_pytorch_tanh"
    cfg.text_config.torch_dtype = "float32"
    cfg.text_config.vocab_size = 257152
    cfg.text_config.use_adarms = False
    cfg.text_config.adarms_cond_dim = None
    cfg.vision_config.intermediate_size = 4304
    cfg.vision_config.projection_dim = 2048
    cfg.vision_config.projector_hidden_act = "gelu_fast"
    cfg.vision_config.torch_dtype = "float32"
    return PaliGemmaForConditionalGeneration(config=cfg)


def load_paligemma(device: torch.device):
    from safetensors.torch import load_file

    model = build_paligemma()
    src = load_file(str(WEIGHT / "model.safetensors"))
    prefix = "paligemma_with_expert.paligemma."
    sub = {k[len(prefix) :]: v for k, v in src.items() if k.startswith(prefix)}
    if not sub:
        raise SystemExit(f"{WEIGHT} 没有 paligemma_with_expert.paligemma.*")
    dest = model.state_dict()
    remapped = {}
    for key, tensor in sub.items():
        target = key
        if target not in dest:
            alt = target.replace(".model.language_model.", ".language_model.")
            if alt in dest:
                target = alt
        if target not in dest and target.startswith("language_model."):
            alt = "model." + target
            if alt in dest:
                target = alt
        if target not in dest and target.startswith("model.language_model."):
            alt = target.replace("model.language_model.", "language_model.", 1)
            if alt in dest:
                target = alt
        if target in dest and tuple(dest[target].shape) == tuple(tensor.shape):
            remapped[target] = tensor
    if (
        "model.language_model.embed_tokens.weight" in dest
        and "model.language_model.embed_tokens.weight" not in remapped
        and "lm_head.weight" in remapped
    ):
        remapped["model.language_model.embed_tokens.weight"] = remapped["lm_head.weight"]
        logging.info("filled embed_tokens from tied lm_head")
    missing, _ = model.load_state_dict(remapped, strict=False)
    logging.info("loaded paligemma tensors=%s missing=%s", len(remapped), len(missing))
    if missing:
        logging.warning("missing sample %s", list(missing)[:8])
    model.to(device=device, dtype=torch.bfloat16)
    return model


class Rows(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = Image.open(row["image"]).convert("RGB")
        return {
            "image": image,
            "prompt": PROMPT.format(task=row["task"]),
            "subtask": row["subtask"],
            "label": row["label"],
        }


def pack(sp, batch: list[dict], device, dtype):
    pixels = []
    ids_rows = []
    lab_rows = []
    for item in batch:
        arr = np.asarray(item["image"].resize((224, 224), Image.BICUBIC), np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        pixels.append(torch.from_numpy(arr).permute(2, 0, 1))
        prefix = list(sp.encode(item["prompt"].strip(), add_bos=True)) + list(sp.encode("\n"))
        suffix = list(sp.encode(item["subtask"].strip(), add_bos=False)) + [int(sp.eos_id())]
        ids = [IMAGE_TOKEN_ID] * N_IMAGE_TOKENS + prefix + suffix
        labs = [-100] * (N_IMAGE_TOKENS + len(prefix)) + suffix
        ids_rows.append(ids)
        lab_rows.append(labs)
    maxlen = max(len(r) for r in ids_rows)
    input_ids = torch.zeros((len(batch), maxlen), dtype=torch.long, device=device)
    labels = torch.full((len(batch), maxlen), -100, dtype=torch.long, device=device)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long, device=device)
    for i, (ids, labs) in enumerate(zip(ids_rows, lab_rows)):
        input_ids[i, : len(ids)] = torch.tensor(ids, device=device)
        labels[i, : len(labs)] = torch.tensor(labs, device=device)
        attn[i, : len(ids)] = 1
    pix = torch.stack(pixels).to(device=device, dtype=dtype)
    return {"pixel_values": pix, "input_ids": input_ids, "attention_mask": attn, "labels": labels}


def norm(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


@torch.no_grad()
def greedy_texts(model, sp, batch: list[dict], device, dtype, max_new: int = 24) -> list[str]:
    """Token-by-token argmax via forward. model.generate hits a PPU SIGSEGV."""
    pixels = []
    seqs: list[list[int]] = []
    starts: list[int] = []
    for item in batch:
        arr = np.asarray(item["image"].resize((224, 224), Image.BICUBIC), np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        pixels.append(torch.from_numpy(arr).permute(2, 0, 1))
        prefix = list(sp.encode(item["prompt"].strip(), add_bos=True)) + list(sp.encode("\n"))
        seqs.append([IMAGE_TOKEN_ID] * N_IMAGE_TOKENS + prefix)
        starts.append(len(seqs[-1]))
    pix = torch.stack(pixels).to(device=device, dtype=dtype)
    eos = int(sp.eos_id())
    finished = [False] * len(batch)
    for _ in range(max_new):
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.zeros((len(batch), maxlen), dtype=torch.long, device=device)
        attn = torch.zeros_like(input_ids)
        for i, seq in enumerate(seqs):
            input_ids[i, : len(seq)] = torch.tensor(seq, device=device)
            attn[i, : len(seq)] = 1
        out = model(pixel_values=pix, input_ids=input_ids, attention_mask=attn)
        pos = attn.sum(dim=1) - 1
        nxt = out.logits[torch.arange(len(batch), device=device), pos].argmax(dim=-1)
        for i, tok in enumerate(nxt.tolist()):
            if finished[i]:
                continue
            seqs[i].append(int(tok))
            if int(tok) == eos:
                finished[i] = True
        if all(finished):
            break
    texts = []
    piece_n = int(sp.GetPieceSize())
    for seq, start in zip(seqs, starts):
        ids = [tok for tok in seq[start:] if tok != eos and 0 <= tok < piece_n]
        texts.append(sp.decode(ids).strip().split("\n")[0].strip())
    return texts


def evaluate(model, sp, rows: list[dict], device, dtype, templates_by_text: dict[str, str]) -> dict | None:
    """Each rank scores its own slice, then rank 0 merges. Greedy forward, not generate."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    shard = rows[rank::world]
    model.eval()
    n = exact = label_hit = 0
    by: dict[str, list[int]] = {}
    examples: list[tuple[str, str]] = []
    ds = Rows(shard)
    for start in range(0, len(shard), 4):
        batch = [ds[i] for i in range(start, min(start + 4, len(shard)))]
        preds = greedy_texts(model, sp, batch, device, dtype)
        for offset, (item, pred) in enumerate(zip(batch, preds)):
            row = shard[start + offset]
            gold = item["subtask"]
            ok = norm(pred) == norm(gold)
            pred_label = templates_by_text.get(norm(pred), "")
            lab = pred_label == item["label"] or ok
            n += 1
            exact += int(ok)
            label_hit += int(lab)
            slot = by.setdefault(row["dataset"] + "/" + item["label"], [0, 0])
            slot[0] += 1
            slot[1] += int(ok)
            if len(examples) < 2:
                examples.append((pred, gold))
        if rank == 0 and n % 80 == 0:
            logging.info("val progress rank0 %s/%s", n, len(shard))
    payload = {"n": n, "exact": exact, "label_hit": label_hit, "by": by, "examples": examples}
    if world > 1:
        gathered: list[dict | None] = [None] * world
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    if rank != 0:
        return None
    n = exact = label_hit = 0
    merged: dict[str, list[int]] = {}
    for part in gathered:
        assert part is not None
        n += part["n"]
        exact += part["exact"]
        label_hit += part["label_hit"]
        for key, (count, hits) in part["by"].items():
            slot = merged.setdefault(key, [0, 0])
            slot[0] += count
            slot[1] += hits
        for pred, gold in part["examples"]:
            logging.info("val pred=%r gold=%r", pred, gold)
    summary = {
        "n": n,
        "exact_match": exact / n if n else 0.0,
        "label_acc": label_hit / n if n else 0.0,
        "by": {k: {"n": v[0], "exact": v[1] / v[0]} for k, v in sorted(merged.items())},
        "world_size": world,
    }
    logging.info("val n=%s exact=%.3f label=%.3f", n, summary["exact_match"], summary["label_acc"])
    return summary


def setup_ddp() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}")
    else:
        device = torch.device("cpu")
    return rank, local, world, device


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import sys

    sys.path.insert(0, "/workspace/openpi/cigai_train/openpi-jiangsuan/src")
    rank, _local, world, device = setup_ddp()
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
    ROOT.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        build_rows()
    if world > 1:
        dist.barrier()
    rows = build_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    logging.info(
        "split train=%s val=%s ratio=%.2f world=%s batch_per_gpu=%s",
        len(train_rows),
        len(val_rows),
        len(train_rows) / max(1, len(val_rows)),
        world,
        BATCH_PER_GPU,
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

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    text_to_label = {}
    for spec in DATASETS:
        for label, text in spec["templates"].items():
            text_to_label[norm(text)] = label
    train_ds = Rows(train_rows)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_PER_GPU,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda b: b,
        num_workers=0,
        drop_last=False,
    )
    opt = torch.optim.AdamW(raw.parameters(), lr=1e-5)
    best = -1.0
    CKPT.mkdir(exist_ok=True)
    logging.info("init weight=%s epochs=%s", WEIGHT, EPOCHS)
    for epoch in range(EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)
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
                logging.info("epoch=%s step=%s loss=%.4f gpus=%s", epoch, steps, total / steps, world)
        if world > 1:
            dist.barrier()
        if rank == 0:
            torch.save(raw.state_dict(), CKPT / "latest.pt")
            logging.info("saved latest before val epoch=%s train_loss=%.4f", epoch, total / max(1, steps))
        metrics = evaluate(raw, sp, val_rows, device, dtype, text_to_label)
        if rank == 0 and metrics is not None:
            metrics["epoch"] = epoch
            metrics["train_loss"] = total / max(1, steps)
            metrics["global_batch"] = BATCH_PER_GPU * world
            (CKPT / f"epoch{epoch}.json").write_text(json.dumps(metrics, indent=2))
            if metrics["exact_match"] > best:
                best = metrics["exact_match"]
                torch.save(raw.state_dict(), CKPT / "best.pt")
                (CKPT / "best_metrics.json").write_text(json.dumps(metrics, indent=2))
                logging.info("saved best exact=%.3f", best)
        if world > 1:
            dist.barrier()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
