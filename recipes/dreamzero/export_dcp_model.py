#!/usr/bin/env python3
"""Export model-only inference weights from a DreamZero 14B DCP checkpoint.

Runs on h100-103 (rlinf-env, CPU only — never touches the training GPUs).
Reads the FSDP2 DCP checkpoint, keeps only the trained DiT/action tensors
(``fsdp_checkpoint.model.action_head.model.*`` — the optimizer state proves
T5/CLIP/VAE are frozen, so those keep coming from the pretrained files),
casts to bf16 and saves a flat ``action_head.model.*`` state dict — the exact
format the 5B ``infer_weights.pt`` used (loaded by infer_tong_dreamzero.py
``_load_weights`` via ``model.load_state_dict(strict=False)``).

Usage:
  /data1/rlinf-env/bin/python export_dcp_model.py \
      --dcp  <ckpt>/actor/dcp_checkpoint \
      --out  /data/train_data/dreamzero-exports/<name>/infer_weights.pt
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

SRC_PREFIX = "fsdp_checkpoint.model.action_head.model."
# strip this much, leaving "action_head.model.*" keys (5B infer_weights format)
STRIP = len("fsdp_checkpoint.model.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dcp", required=True, help="path to actor/dcp_checkpoint")
    p.add_argument("--out", required=True, help="output infer_weights.pt path")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = p.parse_args()

    md = FileSystemReader(args.dcp).read_metadata()
    want = {
        k: v
        for k, v in md.state_dict_metadata.items()
        if k.startswith(SRC_PREFIX) and isinstance(v, TensorStorageMetadata)
    }
    n_total = len(want)
    fp32_bytes = sum(v.size.numel() * v.properties.dtype.itemsize for v in want.values())
    if n_total == 0:
        raise SystemExit(f"no keys with prefix {SRC_PREFIX} in {args.dcp}")
    print(f"[export] {n_total} tensors, fp32 {fp32_bytes / 1e9:.2f} GB", flush=True)

    # Single-process gloo PG so dcp.load is happy without touching NCCL/GPUs.
    dist.init_process_group(
        backend="gloo", init_method="tcp://127.0.0.1:29517", rank=0, world_size=1
    )
    try:
        # Flat dict with dotted keys: dcp's flatten_state_dict maps each leaf
        # key 1:1 onto the metadata key of the same name.
        state = {
            k: torch.empty(v.size, dtype=v.properties.dtype) for k, v in want.items()
        }
        t0 = time.time()
        dcp.load(state, storage_reader=FileSystemReader(args.dcp))
        print(f"[export] dcp.load done in {time.time() - t0:.0f}s", flush=True)
    finally:
        dist.destroy_process_group()

    out_dtype = getattr(torch, args.dtype)
    out = {k[STRIP:]: t.to(out_dtype) for k, t in state.items()}
    del state

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    t0 = time.time()
    torch.save(out, tmp)
    os.replace(tmp, args.out)
    size_g = os.path.getsize(args.out) / 1e9
    print(
        f"[export] wrote {args.out} ({size_g:.2f} GB, {args.dtype}) "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )
    print(f"[export] sample keys: {list(out)[:3]}", flush=True)


if __name__ == "__main__":
    main()
