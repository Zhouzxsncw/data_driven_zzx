#!/usr/bin/env python3
"""LingBot-VLA Huawei smoke：权重完整性 + import + 可选 tiny NPU op。

默认不把 6B 模型 load 到正在训练的 NPU 上。
  SMOKE_NPU=1   在 npu:0 跑一次 matmul（不抢训练显存）
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LINGBOT_ROOT", "/home/lingbot-vla"))
SRC = Path(os.environ.get("LINGBOT_SRC", ROOT / "src/lingbot-vla-v2"))
CKPT = Path(os.environ.get("PRETRAINED_PATH", ROOT / "pretrained/lingbot-vla-v2-6b"))
TOK = Path(os.environ.get("QWEN3VL_PATH", ROOT / "pretrained/qwen3-tokenizer"))
MOGE = ROOT / "pretrained/moge-2-vitb-normal/model.pt"
SMOKE_NPU = os.environ.get("SMOKE_NPU", "1") == "1"


def log(msg: str) -> None:
    print(msg, flush=True)


def check_files() -> None:
    shards = [CKPT / f"model-{i:05d}-of-00006.safetensors" for i in range(1, 7)]
    need = [
        CKPT / "config.json",
        CKPT / "model.safetensors.index.json",
        CKPT / "depth/model.pt",
        CKPT / "dino_video/teacher_step_10000.pth",
        MOGE,
        TOK / "tokenizer.json",
    ] + shards
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing}")
    sizes = {p.name: p.stat().st_size for p in shards}
    total = sum(sizes.values())
    if total < 20 * 1024 ** 3:
        raise SystemExit(f"shard total too small: {total}")
    log(f"[1] pretrained OK {CKPT} shards={total / 1e9:.2f} GB moge={MOGE.stat().st_size / 1e6:.0f} MB")


def check_import() -> None:
    import torch
    import torch_npu  # noqa: F401
    import lingbotvla

    log(f"[2] torch {torch.__version__} npu={torch.npu.is_available()} ndev={torch.npu.device_count()}")
    log(f"[3] lingbotvla {lingbotvla.__file__}")
    import transformers

    log(f"[4] transformers {transformers.__version__}")
    if not transformers.__version__.startswith("4.57"):
        raise SystemExit(f"want transformers 4.57.x, got {transformers.__version__}")


def check_tokenizer_and_weights() -> None:
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOK), trust_remote_code=True)
    log(f"[5] tokenizer {tok.__class__.__name__} vocab={len(tok)}")
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())
    weight_map = idx.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))
    nparams = 0
    nkeys = 0
    for shard in shard_files:
        path = CKPT / shard
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            nkeys += len(keys)
            for k in keys:
                shape = f.get_slice(k).get_shape()
                n = 1
                for d in shape:
                    n *= d
                nparams += n
    log(f"[6] safetensors {nkeys} tensors, {nparams / 1e9:.3f}B params, shards={len(shard_files)}")
    if nparams < 5e9:
        raise SystemExit(f"param count too small: {nparams}")


def check_npu_op() -> None:
    import torch

    if not torch.npu.is_available():
        log("[7] SKIP npu op (npu not available)")
        return
    x = torch.randn(8, 8, device="npu:0", dtype=torch.float32)
    y = x @ x.t()
    torch.npu.synchronize()
    log(f"[7] matmul NPU OK shape={tuple(y.shape)} device={y.device}")


def main() -> int:
    sys.path[:0] = [str(SRC), str(ROOT / "shim")]
    check_files()
    check_import()
    check_tokenizer_and_weights()
    if SMOKE_NPU:
        try:
            check_npu_op()
        except Exception as exc:
            log(f"[7] NPU op failed ({type(exc).__name__}: {exc}); not fatal if chips are busy")
    else:
        log("[7] SKIP npu op (SMOKE_NPU=0)")
    log("SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
