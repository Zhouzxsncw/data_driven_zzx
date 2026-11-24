#!/usr/bin/env bash
# 链接已有 Wan2.2，其余走 ModelScope / hf-mirror。
set -euo pipefail

MOTUS_ROOT="${MOTUS_ROOT:-/home/xk/VLA/Motus}"
PRE="${MOTUS_PRETRAINED:-${MOTUS_ROOT}/pretrained_models}"
LOCAL_WAN22="${LOCAL_WAN22:-/home/xk/VLA/dreamzero_staging/pretrained/Wan2.2-TI2V-5B}"
DOWNLOAD_ROBOTWIN="${DOWNLOAD_ROBOTWIN:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DISABLE_XET=1
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${MOTUS_ROOT}/.modelscope}"
export HF_HOME="${HF_HOME:-${MOTUS_ROOT}/.hf}"

mkdir -p "${PRE}" "${MODELSCOPE_CACHE}" "${HF_HOME}"

pick_python() {
  if [[ -x "${HOME}/miniconda3/envs/motus/bin/python" ]]; then
    echo "${HOME}/miniconda3/envs/motus/bin/python"
  elif [[ -x "${HOME}/anaconda3/envs/motus/bin/python" ]]; then
    echo "${HOME}/anaconda3/envs/motus/bin/python"
  elif [[ -x "${HOME}/miniconda3/envs/fastwam/bin/python" ]]; then
    echo "${HOME}/miniconda3/envs/fastwam/bin/python"
  elif [[ -x "${HOME}/anaconda3/envs/fastwam/bin/python" ]]; then
    echo "${HOME}/anaconda3/envs/fastwam/bin/python"
  else
    command -v python
  fi
}
PYTHON="$(pick_python)"
export PYTHONUNBUFFERED=1
echo "PYTHON=${PYTHON}"
echo "PRE=${PRE}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"

link_or_copy_dir() {
  local src="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -L "$dest" || -d "$dest" ]]; then
    echo "[skip] $dest already exists"
    return 0
  fi
  if [[ -d "$src" ]]; then
    ln -sfn "$src" "$dest"
    echo "[link] $dest -> $src"
  else
    return 1
  fi
}

if [[ -f "${LOCAL_WAN22}/Wan2.2_VAE.pth" ]]; then
  link_or_copy_dir "${LOCAL_WAN22}" "${PRE}/Wan2.2-TI2V-5B" || true
else
  echo "[download] Wan-AI/Wan2.2-TI2V-5B via modelscope"
  "${PYTHON}" - <<PY
from modelscope import snapshot_download
snapshot_download("Wan-AI/Wan2.2-TI2V-5B", local_dir="${PRE}/Wan2.2-TI2V-5B")
PY
fi

echo "[download] Qwen + Motus via hf-mirror (hf_transfer multipart)"
"${PYTHON}" - <<PY
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1")

try:
    import hf_transfer  # noqa: F401
    print("hf_transfer enabled")
except Exception as e:
    print("WARN hf_transfer missing, fallback to huggingface_hub:", e)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download

def complete(dest, marker, min_bytes=1_000_000):
    p = Path(dest) / marker
    return p.exists() and p.stat().st_size > min_bytes

def need(repo, dest, marker, min_bytes=1_000_000, extra_markers=()):
    extras = all((Path(dest) / m).exists() for m in extra_markers)
    if complete(dest, marker, min_bytes) and extras:
        print(f"[skip] {dest} has {marker} ({(Path(dest)/marker).stat().st_size} bytes)")
        return
    print(f"[hf] {repo} -> {dest}", flush=True)
    snapshot_download(repo_id=repo, local_dir=str(dest))

need("Qwen/Qwen3-VL-2B-Instruct", "${PRE}/Qwen3-VL-2B-Instruct", "model.safetensors", min_bytes=3_000_000_000, extra_markers=("tokenizer.json", "preprocessor_config.json"))
need("motus-robotics/Motus_Wan2_2_5B_pretrain", "${PRE}/Motus_Wan2_2_5B_pretrain", "model.safetensors", min_bytes=8_000_000_000)
need("motus-robotics/Motus", "${PRE}/Motus", "mp_rank_00_model_states.pt", min_bytes=10_000_000_000)
if "${DOWNLOAD_ROBOTWIN}" == "1":
    need("motus-robotics/Motus_robotwin2", "${PRE}/Motus_robotwin2", "mp_rank_00_model_states.pt", min_bytes=10_000_000_000)
else:
    print("[skip] Motus_robotwin2 (set DOWNLOAD_ROBOTWIN=1 to fetch)")
PY

echo "[write] ${MOTUS_ROOT}/configs/local.yaml"
"${PYTHON}" - <<PY
from pathlib import Path
src = Path("${MOTUS_ROOT}/configs/robotwin.yaml")
dst = Path("${MOTUS_ROOT}/configs/local.yaml")
text = src.read_text()
repls = {
    'config_path: "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B"':
        'config_path: "${PRE}/Motus_Wan2_2_5B_pretrain"',
    'checkpoint_path: "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B"':
        'checkpoint_path: "${PRE}/Motus_Wan2_2_5B_pretrain"',
    'vae_path: "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"':
        'vae_path: "${PRE}/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"',
    'checkpoint_path: "/share/home/bhz/pretrained_models/Qwen3-VL-2B-Instruct"':
        'checkpoint_path: "${PRE}/Qwen3-VL-2B-Instruct"',
}
for a, b in repls.items():
    if a not in text:
        raise SystemExit(f"pattern not found in robotwin.yaml: {a}")
    text = text.replace(a, b)
# Stage 2 pretrained as SFT start; VLM config_path
if "config_path:" not in text.split("vlm:", 1)[-1].split("action_expert:", 1)[0]:
    text = text.replace(
        'checkpoint_path: "${PRE}/Qwen3-VL-2B-Instruct"\n    precision:',
        'checkpoint_path: "${PRE}/Qwen3-VL-2B-Instruct"\n    config_path: "${PRE}/Qwen3-VL-2B-Instruct"\n    precision:',
        1,
    )
text = text.replace(
    "finetune:\n  checkpoint_path: null",
    "finetune:\n  checkpoint_path: ${PRE}/Motus",
)
text = text.replace('report_to: "wandb"', 'report_to: "tensorboard"')
dst.write_text(text)
print(f"wrote {dst}")
PY

echo "========== pretrained layout =========="
for d in Wan2.2-TI2V-5B Qwen3-VL-2B-Instruct Motus_Wan2_2_5B_pretrain Motus Motus_robotwin2; do
  if [[ -e "${PRE}/${d}" ]]; then
    du -sh "${PRE}/${d}" || true
  fi
done
ls -lh "${PRE}/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
  "${PRE}/Motus/mp_rank_00_model_states.pt" \
  "${PRE}/Motus_Wan2_2_5B_pretrain/model.safetensors" 2>/dev/null || true
echo "OK configs/local.yaml -> ${MOTUS_ROOT}/configs/local.yaml"
