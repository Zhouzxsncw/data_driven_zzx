#!/usr/bin/env python3
"""DM0.5 CPU smoke. Refuses if a PPU node is visible or CUDA sees a device.

Does not copy model.safetensors onto a device. The header is enough to prove
the checkpoint arrived. Importing DM05Config proves the stack, not the card.
"""
from __future__ import annotations

import os

# Hide devices before torch is imported, even if the container was started wrong.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PPU_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/workspace"))
CKPT = Path(os.environ.get("DM05_CKPT", ROOT / "pretrained" / "DM05"))
SRC = ROOT / "src" / "opendm"
EXPECT_FILES = 12
EXPECT_SAFETENSORS = 11_658_431_136


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    visible = sorted(p.name for p in Path("/dev").glob("alixpu*"))
    if visible:
        raise SystemExit(
            "REFUSE /dev/alixpu is visible inside this container: "
            + ",".join(visible)
            + ". run_jiangsuan.sh removes these nodes; do not smoke until they are gone."
        )

    sys.path[:0] = [str(SRC)]
    import torch

    n = torch.cuda.device_count()
    log(f"[1] torch {torch.__version__} device_count={n}")
    if n != 0:
        raise SystemExit(f"REFUSE cuda device_count={n}; smoke must not touch a card")

    files = sorted(p for p in CKPT.iterdir() if p.is_file())
    if len(files) != EXPECT_FILES:
        names = ", ".join(p.name for p in files)
        raise SystemExit(f"REFUSE file count {len(files)} != {EXPECT_FILES}: {names}")
    weight = CKPT / "model.safetensors"
    size = weight.stat().st_size
    if size != EXPECT_SAFETENSORS:
        raise SystemExit(f"REFUSE model.safetensors {size} != {EXPECT_SAFETENSORS}")
    log(f"[2] checkpoint files={len(files)} safetensors={size}")

    from safetensors import safe_open

    with safe_open(str(weight), framework="pt", device="cpu") as handle:
        nkeys = len(handle.keys())
    if nkeys < 1:
        raise SystemExit("REFUSE safetensors header has no tensors")
    log(f"[3] safetensors header keys={nkeys} (weights not copied to a device)")

    raw = json.loads((CKPT / "config.json").read_text())
    from opendm.model.dm05.dm05_arch import DM05Config

    cfg = DM05Config.from_dict(raw) if hasattr(DM05Config, "from_dict") else DM05Config()
    log(f"[4] DM05Config model_type={cfg.model_type} architectures={raw.get('architectures')}")
    log("[5] SKIP device op (no PPU nodes, CUDA_VISIBLE_DEVICES empty)")
    log("SMOKE_PASS")
    log("device_count=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
