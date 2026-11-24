#!/usr/bin/env bash
# 下载 GigaWorld-Policy-0.5 transformer + Wan2.2-TI2V-5B-Diffusers（VAE/scheduler/T5）。
# transformer 骨干用 GWP-0.5 自己的权重，Diffusers Wan 的 transformer/ 默认跳过以省 ~20G。
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1

GWP_ROOT="${GWP_ROOT:-/home/xk/VLA/giga-world-policy}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-/data/giga-world-policy/pretrained}"
GWP_DIR="${PRETRAINED_ROOT}/Giga-World-Policy-0.5"
WAN_DIR="${PRETRAINED_ROOT}/Wan2.2-TI2V-5B-Diffusers"
LOG_FILE="${LOG_FILE:-${GWP_ROOT}/download_pretrained.log}"
FULL_WAN="${FULL_WAN:-0}"  # 设 1 则连 Wan transformer 一起下

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

mkdir -p "$(dirname "$LOG_FILE")" "$GWP_DIR" "$WAN_DIR"
ln -sfn "$PRETRAINED_ROOT" "${GWP_ROOT}/pretrained"

HF_BIN=""
if command -v hf >/dev/null 2>&1; then
  HF_BIN="hf"
elif [[ -x "${HOME}/miniconda3/bin/hf" ]]; then
  HF_BIN="${HOME}/miniconda3/bin/hf"
fi

download_repo() {
  local repo="$1" dest="$2"
  shift 2
  log "download $repo -> $dest"
  if [[ -n "$HF_BIN" ]]; then
    "$HF_BIN" download "$repo" --local-dir "$dest" "$@"
  else
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${repo}", local_dir="${dest}")
PY
  fi
}

log "========== GWP pretrained download =========="
log "HF_ENDPOINT=$HF_ENDPOINT PRETRAINED_ROOT=$PRETRAINED_ROOT FULL_WAN=$FULL_WAN"

need_gwp=0
if [[ ! -f "${GWP_DIR}/diffusion_pytorch_model.safetensors.index.json" ]]; then
  need_gwp=1
elif ! compgen -G "${GWP_DIR}/diffusion_pytorch_model-*.safetensors" >/dev/null; then
  need_gwp=1
fi

if [[ "$need_gwp" -eq 1 ]]; then
  download_repo "open-gigaai/Giga-World-Policy-0.5" "$GWP_DIR" || {
    log "hf-mirror 失败，改走 huggingface.co"
    HF_ENDPOINT=https://huggingface.co download_repo "open-gigaai/Giga-World-Policy-0.5" "$GWP_DIR"
  }
else
  log "skip GWP-0.5 already present"
fi

need_wan=0
if [[ ! -f "${WAN_DIR}/vae/diffusion_pytorch_model.safetensors" ]]; then
  need_wan=1
fi
if [[ ! -f "${WAN_DIR}/scheduler/scheduler_config.json" ]]; then
  need_wan=1
fi
if [[ ! -f "${WAN_DIR}/model_index.json" ]]; then
  need_wan=1
fi

if [[ "$need_wan" -eq 1 ]]; then
  log "Wan Diffusers 走 ModelScope（hf-mirror 会跳到 xet 超时）"
  export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${PRETRAINED_ROOT}/.modelscope}"
  PY_MS="${GWP_PYTHON:-}"
  if [[ -z "$PY_MS" ]]; then
    if [[ -x "${HOME}/miniconda3/envs/gigaworld-policy/bin/python" ]]; then
      PY_MS="${HOME}/miniconda3/envs/gigaworld-policy/bin/python"
    elif [[ -x "${HOME}/miniconda3/envs/fastwam/bin/python" ]]; then
      PY_MS="${HOME}/miniconda3/envs/fastwam/bin/python"
    else
      PY_MS="python3"
    fi
  fi
  "$PY_MS" - <<PY
from modelscope.hub.snapshot_download import snapshot_download
ignore = [r"assets/.*", r"examples/.*"]
if "${FULL_WAN}" != "1":
    ignore.append(r"transformer/.*")
snapshot_download(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    local_dir="${WAN_DIR}",
    ignore_file_pattern=ignore,
)
print("modelscope wan done")
PY
else
  log "skip Wan Diffusers VAE already present"
fi

log "========== layout =========="
du -sh "$GWP_DIR" "$WAN_DIR" 2>/dev/null || true
echo "--- GWP-0.5 ---"
ls -lh "$GWP_DIR" | tee -a "$LOG_FILE"
echo "--- Wan Diffusers ---"
ls -lh "$WAN_DIR" | tee -a "$LOG_FILE"
ls -lh "$WAN_DIR/vae" "$WAN_DIR/scheduler" 2>/dev/null | tee -a "$LOG_FILE" || true

python3 - <<PY
from pathlib import Path
gwp = Path("${GWP_DIR}")
wan = Path("${WAN_DIR}")
need = [
    gwp / "diffusion_pytorch_model.safetensors.index.json",
    wan / "model_index.json",
    wan / "vae" / "diffusion_pytorch_model.safetensors",
    wan / "scheduler" / "scheduler_config.json",
]
missing = [str(p) for p in need if not p.exists()]
shards = list(gwp.glob("diffusion_pytorch_model-*.safetensors"))
if missing:
    raise SystemExit("MISSING:\\n" + "\\n".join(missing))
if not shards:
    raise SystemExit("MISSING GWP shards under " + str(gwp))
print(f"OK gwp_shards={len(shards)} vae={ (wan/'vae'/'diffusion_pytorch_model.safetensors').stat().st_size }")
PY

log "========== download done =========="
echo "GWP05_TRANSFORMER_PRETRAINED=$GWP_DIR"
echo "GWP_WAN_PRETRAINED=$WAN_DIR"
