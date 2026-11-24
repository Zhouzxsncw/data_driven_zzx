#!/usr/bin/env python3
"""Cosmos3 就绪测试：权重完整性 + SDPA + 可选单卡 tiny attn。

  SMOKE_DEV=1 CUDA_VISIBLE_DEVICES=0  才碰一张卡
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", os.environ.get("COSMOS3_ROOT", "/workspace")))
SRC = Path(os.environ.get("SRC", ROOT / "src/cosmos-framework"))
MODEL = Path(os.environ.get("COSMOS3_MODEL", ROOT / "pretrained/Cosmos3-Nano"))
SMOKE_DEV = os.environ.get("SMOKE_DEV", "0") == "1"
EXPECT_FILES = int(os.environ.get("EXPECT_FILES", "205"))
EXPECT_BYTES = int(os.environ.get("EXPECT_BYTES", "34986980007"))


def log(msg: str) -> None:
    print(msg, flush=True)


def du_files(p: Path) -> tuple[int, int]:
    n = b = 0
    for f in p.rglob("*"):
        if f.is_file() and not f.is_symlink():
            n += 1
            b += f.stat().st_size
        elif f.is_symlink() and f.exists() and f.is_file():
            n += 1
            b += f.stat().st_size
    return n, b


def check_weights() -> None:
    if not MODEL.is_dir():
        raise SystemExit(f"missing model dir {MODEL}")
    n, b = du_files(MODEL)
    log(f"[w1] {MODEL} files={n} bytes={b}")
    if n < EXPECT_FILES:
        raise SystemExit(f"file count {n} < {EXPECT_FILES}")
    if abs(b - EXPECT_BYTES) > 2_000_000:  # allow tar vs rsync slack
        raise SystemExit(f"byte size {b} != {EXPECT_BYTES} (±2MB)")
    cfg = json.loads((MODEL / "config.json").read_text())
    log(f"[w2] config.json keys={len(cfg)} model_type={cfg.get('model_type') or cfg.get('_class_name')}")
    idx = MODEL / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    if not idx.is_file():
        raise SystemExit(f"missing {idx}")
    weight_map = json.loads(idx.read_text()).get("weight_map", {})
    shards = sorted(set(weight_map.values()))
    missing = [s for s in shards if not (MODEL / "transformer" / s).is_file()]
    if missing:
        raise SystemExit(f"missing transformer shards {missing}")
    from safetensors import safe_open

    sample = MODEL / "transformer" / shards[0]
    with safe_open(str(sample), framework="pt") as f:
        keys = list(f.keys())
    log(f"[w3] shard0 {sample.name} tensors={len(keys)}")
    if len(keys) < 10:
        raise SystemExit("shard too empty")


def check_sdpa() -> None:
    sys.path[:0] = [str(SRC), str(ROOT / "shim")]
    import torch
    from cosmos_framework.model.attention.frontend import BACKEND_MAP, attention

    log(f"[s1] torch {torch.__version__} cuda={torch.cuda.is_available()} backends={sorted(BACKEND_MAP)}")
    if "sdpa" not in BACKEND_MAP:
        raise SystemExit("sdpa not registered")
    q = torch.randn(2, 16, 8, 64, dtype=torch.bfloat16)
    k = torch.randn(2, 16, 8, 64, dtype=torch.bfloat16)
    v = torch.randn(2, 16, 8, 64, dtype=torch.bfloat16)
    out = attention(q, k, v, backend="sdpa")
    if tuple(out.shape) != (2, 16, 8, 64):
        raise SystemExit(f"bad sdpa shape {tuple(out.shape)}")
    log(f"[s2] cpu sdpa OK shape={tuple(out.shape)} mean={float(out.float().mean()):.4f}")

    if not SMOKE_DEV:
        log("[s3] SKIP device (SMOKE_DEV=0)")
        return
    if not torch.cuda.is_available():
        log("[s3] SKIP device (no cuda)")
        return
    dev = torch.device("cuda:0")
    qd, kd, vd = q.to(dev), k.to(dev), v.to(dev)
    od = attention(qd, kd, vd, backend="sdpa")
    log(f"[s3] device sdpa OK {od.device} shape={tuple(od.shape)}")


def main() -> int:
    check_weights()
    check_sdpa()
    log("TEST_PASS cosmos3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
