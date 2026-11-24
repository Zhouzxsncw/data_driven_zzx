#!/usr/bin/env bash
# GigaWorld-Policy-0.5 环境：gpu-master / RTX PRO 6000 Blackwell (sm_120)
# 官方 README 写 cu126；本机 driver 13.2 / Blackwell，跟 FastWAM 一样走 cu128。
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET=1
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export DS_BUILD_AIO=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_OPS="${DS_BUILD_OPS:-0}"

CONDA_ENV="${CONDA_ENV:-gigaworld-policy}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GWP_ROOT="${GWP_ROOT:-/home/xk/VLA/giga-world-policy}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-/data/giga-world-policy/pretrained}"
LOG_FILE="${LOG_FILE:-${GWP_ROOT}/setup_env.log}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

mkdir -p "$(dirname "$LOG_FILE")" "$GWP_ROOT" "$PRETRAINED_ROOT"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "找不到 conda（~/miniconda3 或 ~/anaconda3）" >&2
  exit 1
fi

if [[ ! -d "${GWP_ROOT}/.git" ]]; then
  echo "代码不存在: ${GWP_ROOT} （先 git clone open-gigaai/giga-world-policy）" >&2
  exit 1
fi

log "========== GigaWorld-Policy-0.5 setup =========="
log "GWP_ROOT=$GWP_ROOT CONDA_ENV=$CONDA_ENV PRETRAINED_ROOT=$PRETRAINED_ROOT"
log "PIP_INDEX_URL=$PIP_INDEX_URL HF_ENDPOINT=$HF_ENDPOINT TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  log "creating conda env ${CONDA_ENV} python=3.11"
  conda create -y -n "${CONDA_ENV}" python=3.11 pip
  conda install -y -n "${CONDA_ENV}" -c conda-forge ffmpeg || true
fi
# shellcheck disable=SC1091
set +u
conda activate "${CONDA_ENV}"
set -u

export PYTHONNOUSERSITE=1
python -m pip install -U pip setuptools wheel packaging ninja
log "python=$(python --version) exe=$(which python)"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log "torch already ok: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
else
  log "install torch 2.7.1+cu128 from download.pytorch.org (Blackwell)"
  python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
fi

python - <<'PY' | tee -a "$LOG_FILE"
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"gpu0={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
PY

REQ="${GWP_ROOT}/requirements.txt"
REQ_FILTERED="$(mktemp)"
# 官方同时钉了 diffusers==0.36.0 和 lerobot==0.4.4，但 lerobot 0.4.4 声明 diffusers<0.36。
# 推理/训练需要 0.36 的 Wan pipeline，lerobot 单独 --no-deps。
# xformers 在 Blackwell 上可能没有对应轮子。
grep -vE '^[[:space:]]*(xformers|lerobot)' "$REQ" > "$REQ_FILTERED"

log "install python deps from Aliyun PyPI (skip xformers / lerobot pin conflict)"
python -m pip install -r "$REQ_FILTERED"
rm -f "$REQ_FILTERED"

log "lerobot==0.4.4 --no-deps (keep diffusers 0.36.0)"
python -m pip install "lerobot==0.4.4" --no-deps
# lerobot 运行时常用依赖（不含 diffusers）
python -m pip install "datasets>=4.0.0,<5.0.0" "jsonlines" || log "WARN extra lerobot deps skipped"

# Blackwell 上 xformers 0.0.31 经常没轮子且很慢；默认跳过，需要时 XFORMERS=1
if [[ "${XFORMERS:-0}" == "1" ]] && python -m pip install "xformers==0.0.31.post1"; then
  log "xformers ok"
else
  log "skip xformers (set XFORMERS=1 to try). 推理走 torch.compile / SDPA"
fi

log "editable install third_party (no-deps)"
python -m pip install --no-deps -e "${GWP_ROOT}/third_party/giga-train"
python -m pip install --no-deps -e "${GWP_ROOT}/third_party/giga-models"
python -m pip install --no-deps -e "${GWP_ROOT}/third_party/giga-datasets"

log "========== import check =========="
python - <<PY | tee -a "$LOG_FILE"
import os, sys
root = "${GWP_ROOT}"
sys.path[:0] = [root]
os.environ["GIGA_MODELS_LIGHT_IMPORT"] = "1"
import importlib
for m in ["torch", "torchvision", "accelerate", "diffusers", "transformers", "safetensors", "huggingface_hub"]:
    mod = importlib.import_module(m)
    print(f"OK {m} {getattr(mod, '__version__', '')}")
try:
    import xformers
    print("OK xformers", getattr(xformers, "__version__", ""))
except Exception as e:
    print("WARN xformers:", e)
import world_action_model
print("OK world_action_model", getattr(world_action_model, "__file__", ""))
print("gpu", __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else None)
PY

cat > "${GWP_ROOT}/env.sh" <<EOF
# source ${GWP_ROOT}/env.sh
unalias -a 2>/dev/null || true
if [[ -f "\${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "\${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "\${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
set +u
conda activate ${CONDA_ENV}
set +u
export GWP_ROOT="${GWP_ROOT}"
export PRETRAINED_ROOT="${PRETRAINED_ROOT}"
export GWP05_TRANSFORMER_PRETRAINED="${PRETRAINED_ROOT}/Giga-World-Policy-0.5"
export GWP_WAN_PRETRAINED="${PRETRAINED_ROOT}/Wan2.2-TI2V-5B-Diffusers"
export BASE_MODEL="\${GWP_WAN_PRETRAINED}"
export CHECKPOINT="\${GWP05_TRANSFORMER_PRETRAINED}"
export HF_ENDPOINT="${HF_ENDPOINT}"
export HF_HUB_DISABLE_XET=1
export PIP_INDEX_URL="${PIP_INDEX_URL}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}"
export HF_HOME="\${HF_HOME:-${PRETRAINED_ROOT}/.hf}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
export GIGA_MODELS_LIGHT_IMPORT=1
export PYTHONPATH="${GWP_ROOT}:${GWP_ROOT}/third_party/giga-train:${GWP_ROOT}/third_party/giga-models:${GWP_ROOT}/third_party/giga-datasets:\${PYTHONPATH:-}"
cd "\${GWP_ROOT}"
echo "python=\$(which python) \$(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo "GWP05=\${GWP05_TRANSFORMER_PRETRAINED}"
echo "WAN=\${GWP_WAN_PRETRAINED}"
EOF

cp -a "${GWP_ROOT}/env.sh" "${HERE}/env.sh"
log "wrote ${GWP_ROOT}/env.sh and ${HERE}/env.sh"
log "========== setup done =========="
