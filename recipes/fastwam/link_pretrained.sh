#!/usr/bin/env bash
# 把 Wan2.2-TI2V-5B / T5 tokenizer 放到 FastWAM checkpoints/（国内 ModelScope，或本机已有权重）。
set -euo pipefail

FASTWAM_ROOT="${FASTWAM_ROOT:-/home/xk/VLA/FastWAM}"
CKPT="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_ROOT}/checkpoints}"
LOCAL_WAN22="${LOCAL_WAN22:-/home/xk/VLA/dreamzero_staging/pretrained/Wan2.2-TI2V-5B}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${FASTWAM_ROOT}/.modelscope}"

mkdir -p "${CKPT}/Wan-AI" "${CKPT}/DiffSynth-Studio"

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

echo "CKPT=$CKPT LOCAL_WAN22=$LOCAL_WAN22"

if [[ -f "${LOCAL_WAN22}/diffusion_pytorch_model.safetensors.index.json" ]]; then
  link_or_copy_dir "${LOCAL_WAN22}" "${CKPT}/Wan-AI/Wan2.2-TI2V-5B" || true
else
  echo "[download] Wan-AI/Wan2.2-TI2V-5B via modelscope"
  python - <<PY
from modelscope import snapshot_download
snapshot_download("Wan-AI/Wan2.2-TI2V-5B", local_dir="${CKPT}/Wan-AI/Wan2.2-TI2V-5B")
PY
fi

# FastWAM tokenizer 路径：checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
TOK_DEST="${CKPT}/Wan-AI/Wan2.1-T2V-1.3B"
mkdir -p "${TOK_DEST}"
if [[ -d "${CKPT}/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl" ]]; then
  ln -sfn "${CKPT}/Wan-AI/Wan2.2-TI2V-5B/google" "${TOK_DEST}/google"
  echo "[link] tokenizer ${TOK_DEST}/google -> Wan2.2 google/umt5-xxl"
elif [[ ! -d "${TOK_DEST}/google/umt5-xxl" ]]; then
  echo "[download] Wan-AI/Wan2.1-T2V-1.3B tokenizer via modelscope"
  python - <<PY
from modelscope import snapshot_download
snapshot_download(
    "Wan-AI/Wan2.1-T2V-1.3B",
    local_dir="${TOK_DEST}",
    allow_file_pattern="google/umt5-xxl/*",
)
PY
fi

need_dit=$(compgen -G "${CKPT}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors" >/dev/null && echo 0 || echo 1)
need_t5=$([[ -f "${CKPT}/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" ]] && echo 0 || echo 1)
need_vae=$([[ -f "${CKPT}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" ]] && echo 0 || echo 1)
echo "dit_ok=$((1-need_dit)) t5_ok=$((1-need_t5)) vae_ok=$((1-need_vae))"
if [[ "$need_dit" -ne 0 || "$need_t5" -ne 0 || "$need_vae" -ne 0 ]]; then
  echo "pretrained 不完整，尝试 ModelScope 补齐 Wan-AI/Wan2.2-TI2V-5B" >&2
  python - <<PY
from modelscope import snapshot_download
snapshot_download("Wan-AI/Wan2.2-TI2V-5B", local_dir="${CKPT}/Wan-AI/Wan2.2-TI2V-5B")
PY
fi

echo "========== pretrained layout =========="
du -sh "${CKPT}/Wan-AI/Wan2.2-TI2V-5B" "${TOK_DEST}" 2>/dev/null || true
ls -lh "${CKPT}/Wan-AI/Wan2.2-TI2V-5B" | head
