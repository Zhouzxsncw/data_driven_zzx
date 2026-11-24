#!/usr/bin/env python3
"""GR00T 就绪测试：权重完整性 + import。SMOKE_DEV=1 才碰一张卡。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", os.environ.get("GR00T_ROOT", "/workspace")))
SRC = Path(os.environ.get("SRC", ROOT / "src/Isaac-GR00T"))
CKPT = Path(os.environ.get("GR00T_CKPT", ROOT / "pretrained/GR00T-N1.7-3B"))
BB = Path(os.environ.get("GR00T_BACKBONE_PATH", ROOT / "pretrained/Cosmos-Reason2-2B"))
SMOKE_DEV = os.environ.get("SMOKE_DEV", "0") == "1"


def log(msg: str) -> None:
    print(msg, flush=True)


def need(p: Path) -> None:
    if not p.exists():
        raise SystemExit(f"missing {p}")


def check_weights() -> None:
    need(CKPT / "config.json")
    cfg = json.loads((CKPT / "config.json").read_text())
    log(f"[w1] GR00T-N1.7 {CKPT} model_type={cfg.get('model_type')}")
    idx = CKPT / "model.safetensors.index.json"
    need(idx)
    weight_map = json.loads(idx.read_text()).get("weight_map", {})
    shards = sorted(set(weight_map.values()))
    missing = [s for s in shards if not (CKPT / s).is_file()]
    if missing:
        raise SystemExit(f"missing ckpt shards {missing}")
    total = sum((CKPT / s).stat().st_size for s in shards)
    log(f"[w2] ckpt shards={len(shards)} bytes={total}")
    if total < 6 * 1024**3:
        raise SystemExit(f"ckpt too small {total}")

    need(BB / "config.json")
    bb_cfg = json.loads((BB / "config.json").read_text())
    st = BB / "model.safetensors"
    need(st)
    log(f"[w3] backbone {BB.name} model_type={bb_cfg.get('model_type')} bytes={st.stat().st_size}")
    if st.stat().st_size < 4 * 1024**3:
        raise SystemExit("backbone too small")

    from safetensors import safe_open

    with safe_open(str(CKPT / shards[0]), framework="pt") as f:
        n = len(f.keys())
    log(f"[w4] ckpt shard0 tensors={n}")
    if n < 10:
        raise SystemExit("ckpt shard empty")


def check_import() -> None:
    sys.path[:0] = [str(SRC), str(ROOT / "shim")]
    import torch
    import gr00t

    log(f"[i1] torch {torch.__version__} cuda={torch.cuda.is_available()}")
    log(f"[i2] gr00t {gr00t.__file__}")
    try:
        import flash_attn  # noqa: F401

        log("[i3] flash_attn present")
    except ImportError:
        log("[i3] flash_attn absent (sdpa fallback)")

    if not SMOKE_DEV:
        log("[i4] SKIP device (SMOKE_DEV=0)")
        return
    if hasattr(torch, "npu") and torch.npu.is_available():
        x = torch.randn(8, 8, device="npu:0")
        y = x @ x.t()
        torch.npu.synchronize()
        log(f"[i4] npu matmul OK {y.device}")
        return
    if torch.cuda.is_available():
        x = torch.randn(8, 8, device="cuda:0")
        y = x @ x.t()
        log(f"[i4] cuda matmul OK {y.device}")
        return
    log("[i4] SKIP device (none)")


def main() -> int:
    check_weights()
    check_import()
    log("TEST_PASS gr00t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
