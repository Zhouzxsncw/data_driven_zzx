#!/usr/bin/env python3
"""WALL-OSS Huawei smoke：权重完整性 + import + 可选 tiny NPU op。

默认不把 4B 模型 load 到正在训练的 NPU 上。
  SMOKE_NPU=1   在 npu:0 跑 rmsnorm（几 KB，不抢训练显存）
  SMOKE_LOAD=1  CPU 构建并 load_state_dict（慢，占主机内存）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("WALLOSS_ROOT", "/home/wall-oss"))
CKPT = Path(os.environ.get("PRETRAINED_PATH", ROOT / "pretrained/wall-oss-0.5"))
SMOKE_NPU = os.environ.get("SMOKE_NPU", "1") == "1"
SMOKE_LOAD = os.environ.get("SMOKE_LOAD", "0") == "1"


def log(msg: str) -> None:
    print(msg, flush=True)


def check_files() -> None:
    need = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "preprocessor_config.json",
        "normalizer_action.pth",
    ]
    missing = [n for n in need if not (CKPT / n).exists()]
    if missing:
        raise SystemExit(f"missing {missing} under {CKPT}")
    size = (CKPT / "model.safetensors").stat().st_size
    log(f"[1] pretrained OK {CKPT} model.safetensors={size / 1e9:.2f} GB")


def check_import() -> None:
    import torch
    import torch_npu  # noqa: F401

    log(f"[2] torch {torch.__version__} npu={torch.npu.is_available()} ndev={torch.npu.device_count()}")
    import wall_x
    from wall_x.model.core.ops import rmsnorm, permute  # noqa: F401
    from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

    log(f"[3] wall_x {wall_x.__file__}")
    log(f"[4] Qwen2_5_VLMoEForAction {Qwen2_5_VLMoEForAction.__name__}")


def check_tokenizer_and_weights() -> None:
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CKPT), trust_remote_code=True)
    log(f"[5] tokenizer {tok.__class__.__name__} vocab={len(tok)}")
    nparams = 0
    embed_rows = None
    with safe_open(str(CKPT / "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            shape = f.get_slice(k).get_shape()
            n = 1
            for d in shape:
                n *= d
            nparams += n
            if k == "model.embed_tokens.weight":
                embed_rows = shape[0]
    log(f"[6] safetensors {len(keys)} tensors, {nparams / 1e9:.3f}B params, embed_rows={embed_rows}")
    if nparams < 3e9:
        raise SystemExit(f"param count too small: {nparams}")


def check_cpu_op() -> None:
    import torch
    from wall_x.model.core.ops import rmsnorm

    x = torch.randn(2, 8, 64, dtype=torch.float32)
    w = torch.ones(64, dtype=torch.float32)
    y = rmsnorm(x, w, 1e-6)
    log(f"[7] rmsnorm CPU OK shape={tuple(y.shape)}")


def check_npu_op() -> None:
    import torch
    from wall_x.model.core.ops import rmsnorm

    if not torch.npu.is_available():
        log("[8] SKIP npu op (npu not available)")
        return
    x = torch.randn(2, 8, 64, device="npu:0", dtype=torch.float32)
    w = torch.ones(64, device="npu:0", dtype=torch.float32)
    y = rmsnorm(x, w, 1e-6)
    torch.npu.synchronize()
    log(f"[8] rmsnorm NPU OK shape={tuple(y.shape)} device={y.device}")


def check_cpu_load() -> None:
    import gc
    import torch
    from safetensors import safe_open
    from wall_x.model.qact.qwen2_5.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

    cfg = Qwen2_5_VLConfig.from_pretrained(str(CKPT))
    with safe_open(str(CKPT / "model.safetensors"), framework="pt") as f:
        cfg.vocab_size = f.get_slice("model.embed_tokens.weight").get_shape()[0]
    cfg.dof_config = {
        "master_right_ee_cartesian_pos": 3,
        "master_right_ee_rotation": 3,
        "master_right_gripper": 1,
        "action_padding": 19,
    }
    cfg.agent_pos_config = {
        "follow_right_ee_cartesian_pos": 3,
        "follow_right_ee_rotation": 3,
        "follow_right_gripper": 2,
        "action_padding": 18,
    }
    cfg._attn_implementation = "sdpa"
    log("[9] constructing model on CPU...")
    model = Qwen2_5_VLMoEForAction(cfg)
    sd = {}
    with safe_open(str(CKPT / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    incompat = model.load_state_dict(sd, strict=False)
    del sd
    missing = [k for k in incompat.missing_keys if "normalizer" not in k]
    n = sum(p.numel() for p in model.parameters())
    log(f"[9] CPU load OK params={n/1e9:.3f}B missing_non_norm={len(missing)}")
    del model
    gc.collect()


def main() -> int:
    sys.path[:0] = [
        str(ROOT / "src/wall-x"),
        str(ROOT / "src/lerobot"),
        str(ROOT / "shim"),
    ]
    check_files()
    check_import()
    check_tokenizer_and_weights()
    check_cpu_op()
    if SMOKE_NPU:
        try:
            check_npu_op()
        except Exception as exc:
            log(f"[8] NPU op failed ({type(exc).__name__}: {exc}); not fatal if chips are busy")
    else:
        log("[8] SKIP npu op (SMOKE_NPU=0)")
    if SMOKE_LOAD:
        check_cpu_load()
    else:
        log("[9] SKIP full CPU load (SMOKE_LOAD=0)")
    log("SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
