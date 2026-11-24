#!/usr/bin/env python3
"""Cosmos3 smoke：import + SDPA 后端注册。默认不占训练卡。

  SMOKE_DEV=1  才在 cuda:0 / npu:0 跑一次 tiny attention
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", os.environ.get("COSMOS3_ROOT", "/workspace")))
SRC = Path(os.environ.get("SRC", ROOT / "src/cosmos-framework"))
SMOKE_DEV = os.environ.get("SMOKE_DEV", "0") == "1"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    sys.path[:0] = [str(SRC), str(ROOT / "shim"), str(ROOT / "code" / "huawei")]
    try:
        import sitecustomize  # noqa: F401
    except Exception:
        pass
    import torch
    import cosmos_framework
    from cosmos_framework.model.attention.frontend import BACKEND_MAP, attention

    log(f"[1] torch {torch.__version__} cuda={torch.cuda.is_available()}")
    log(f"[2] cosmos_framework {cosmos_framework.__file__}")
    log(f"[3] backends {sorted(BACKEND_MAP)}")
    if "sdpa" not in BACKEND_MAP:
        raise SystemExit(f"sdpa not registered: {sorted(BACKEND_MAP)}")

    q = torch.randn(1, 8, 4, 32, dtype=torch.bfloat16)
    k = torch.randn(1, 8, 4, 32, dtype=torch.bfloat16)
    v = torch.randn(1, 8, 4, 32, dtype=torch.bfloat16)
    out = attention(q, k, v, backend="sdpa")
    log(f"[4] cpu sdpa OK shape={tuple(out.shape)}")

    if SMOKE_DEV:
        dev = None
        if hasattr(torch, "npu") and torch.npu.is_available():
            dev = torch.device("npu:0")
        elif torch.cuda.is_available():
            dev = torch.device("cuda:0")
        if dev is None:
            log("[5] SKIP device op (no npu/cuda)")
        else:
            qd, kd, vd = q.to(dev), k.to(dev), v.to(dev)
            od = attention(qd, kd, vd, backend="sdpa")
            log(f"[5] device sdpa OK {od.device} shape={tuple(od.shape)}")
    else:
        log("[5] SKIP device op (SMOKE_DEV=0, training cards left alone)")
    log("SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
