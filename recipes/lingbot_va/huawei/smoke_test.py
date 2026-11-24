#!/usr/bin/env python3
"""LingBot-VA 华为 smoke：基座文件 + import。不把 transformer / VAE load 到 NPU。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

CODE = Path(os.environ.get("LINGBOT_VA_CODE", "/home/lingbot-va/src/lingbot-va"))
BASE = Path(os.environ.get("LINGBOT_VA_BASE", "/home/lingbot-va/pretrained/lingbot-va-base"))


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    need = [
        BASE / "vae/diffusion_pytorch_model.safetensors",
        BASE / "vae/config.json",
        BASE / "text_encoder/model-00001-of-00003.safetensors",
        BASE / "text_encoder/model-00002-of-00003.safetensors",
        BASE / "text_encoder/model-00003-of-00003.safetensors",
        BASE / "text_encoder/config.json",
        BASE / "tokenizer/tokenizer.json",
    ]
    missing = [str(p) for p in need if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise SystemExit(f"missing {missing}")
    vae = (BASE / "vae/diffusion_pytorch_model.safetensors").stat().st_size
    te = sum((BASE / "text_encoder" / f"model-{i:05d}-of-00003.safetensors").stat().st_size for i in range(1, 4))
    if vae < 2 * 1024**3 or te < 8 * 1024**3:
        raise SystemExit(f"weights too small vae={vae} text_encoder={te}")
    log(f"[1] pretrained OK {BASE} vae={vae / 1e9:.2f}GB text_encoder={te / 1e9:.2f}GB")
    sys.path.insert(0, str(CODE))
    import wan_va

    log(f"[2] wan_va {wan_va.__file__}")
    log("SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
