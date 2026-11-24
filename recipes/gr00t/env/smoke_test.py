#!/usr/bin/env python3
"""GR00T smoke：import。默认不占训练卡。Qwen3 主干已有 flash→SDPA fallback。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", os.environ.get("GR00T_ROOT", "/workspace")))
SRC = Path(os.environ.get("SRC", ROOT / "src/Isaac-GR00T"))
SMOKE_DEV = os.environ.get("SMOKE_DEV", "0") == "1"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    sys.path[:0] = [str(SRC), str(ROOT / "shim")]
    import torch
    import gr00t

    log(f"[1] torch {torch.__version__} cuda={torch.cuda.is_available()}")
    log(f"[2] gr00t {gr00t.__file__}")
    try:
        import flash_attn  # noqa: F401
        log("[3] flash_attn present (unexpected on PPU/NPU, OK if NVIDIA)")
    except ImportError:
        log("[3] flash_attn absent → Qwen3 backbone will use sdpa")

    if SMOKE_DEV:
        if hasattr(torch, "npu") and torch.npu.is_available():
            x = torch.randn(8, 8, device="npu:0")
            y = x @ x.t()
            torch.npu.synchronize()
            log(f"[4] npu matmul OK {y.device}")
        elif torch.cuda.is_available():
            x = torch.randn(8, 8, device="cuda:0")
            y = x @ x.t()
            log(f"[4] cuda matmul OK {y.device}")
        else:
            log("[4] SKIP device op")
    else:
        log("[4] SKIP device op (SMOKE_DEV=0)")
    log("SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
