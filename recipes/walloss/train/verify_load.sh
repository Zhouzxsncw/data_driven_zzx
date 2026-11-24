#!/bin/bash
# 容器内：确认能 import 并读到 0.5 权重（不跑训练）。
set -euo pipefail
ROOT="${ROOT:-/workspace}"
CKPT="${CKPT:-${ROOT}/pretrained/wall-oss-0.5}"
VENV="${VENV:-${ROOT}/venv}"

set +u
source /usr/local/PPU_SDK/envsetup.sh
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
set -u

python - << PY
import gc
import torch
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer
from wall_x.model.qact.qwen2_5.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

ckpt = Path("${CKPT}")
print("ndev", torch.cuda.device_count(),
      "name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert ckpt.is_dir() and (ckpt / "model.safetensors").exists(), ckpt

tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
print("tokenizer", tok.__class__.__name__, "vocab", len(tok))

st_path = str(ckpt / "model.safetensors")
nparams = 0
embed_rows = None
with safe_open(st_path, framework="pt") as f:
    keys = list(f.keys())
    for k in keys:
        shape = f.get_slice(k).get_shape()
        n = 1
        for d in shape:
            n *= d
        nparams += n
        if k == "model.embed_tokens.weight":
            embed_rows = shape[0]
print(f"safetensors {len(keys)} tensors, {nparams/1e9:.3f}B params, embed_rows={embed_rows}")

cfg = Qwen2_5_VLConfig.from_pretrained(str(ckpt))
cfg.vocab_size = int(embed_rows or cfg.vocab_size)
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
print("config layers", cfg.num_hidden_layers, "hidden", cfg.hidden_size,
      "experts", cfg.num_experts, "vocab", cfg.vocab_size,
      "vision_inter", cfg.vision_config.intermediate_size)

print("constructing model on CPU...")
model = Qwen2_5_VLMoEForAction(cfg)
sd = {}
with safe_open(st_path, framework="pt") as f:
    for k in f.keys():
        sd[k] = f.get_tensor(k)
incompat = model.load_state_dict(sd, strict=False)
del sd
missing = [k for k in incompat.missing_keys if "normalizer" not in k]
unexpected = [k for k in incompat.unexpected_keys if "normalizer" not in k]
print("missing (non-normalizer)", len(missing), missing[:8])
print("unexpected (non-normalizer)", len(unexpected), unexpected[:8])
print("normalizer extras", sum(1 for k in incompat.unexpected_keys if "normalizer" in k))

model = model.to(dtype=torch.bfloat16)
if torch.cuda.is_available():
    model = model.to("cuda:0")
    t = next(model.parameters())
    print("device", t.device, "dtype", t.dtype, "cuda_mem_gb",
          torch.cuda.max_memory_allocated(0) / 1e9)
n = sum(p.numel() for p in model.parameters())
print("loaded", type(model).__name__, f"{n/1e9:.3f}B params")
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("[OK] verify finished")
PY
