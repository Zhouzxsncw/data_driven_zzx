#!/usr/bin/env bash
# Motus 训练/推理环境：gpu-master / gpu-node1（Blackwell sm_120）
# 国内源：阿里云 PyPI；GitHub 失败则 gitclone.com。
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET=1
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export DS_BUILD_AIO=0
export DS_BUILD_SPARSE_ATTN=0
export MODELSCOPE_DOMAIN="${MODELSCOPE_DOMAIN:-www.modelscope.cn}"

CONDA_ENV="${CONDA_ENV:-motus}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTUS_ROOT="${MOTUS_ROOT:-/home/xk/VLA/Motus}"
GIT_URL="${MOTUS_GIT_URL:-https://github.com/thu-ml/Motus.git}"
GIT_MIRROR="${MOTUS_GIT_MIRROR:-https://gitclone.com/github.com/thu-ml/Motus.git}"
LOG_FILE="${LOG_FILE:-${HERE}/setup_env.log}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$MOTUS_ROOT")"

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

log "========== Motus setup =========="
log "MOTUS_ROOT=$MOTUS_ROOT CONDA_ENV=$CONDA_ENV PIP_INDEX_URL=$PIP_INDEX_URL HF_ENDPOINT=$HF_ENDPOINT"

if [[ ! -f "${MOTUS_ROOT}/requirements.txt" ]]; then
  log "clone Motus -> ${MOTUS_ROOT}"
  mkdir -p "$(dirname "$MOTUS_ROOT")"
  if ! git clone --depth 1 "${GIT_URL}" "${MOTUS_ROOT}"; then
    log "GitHub clone failed, try gitclone.com"
    rm -rf "${MOTUS_ROOT}"
    git clone --depth 1 "${GIT_MIRROR}" "${MOTUS_ROOT}"
  fi
else
  log "Motus repo already at ${MOTUS_ROOT}"
fi

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  log "creating conda env ${CONDA_ENV} python=3.10"
  conda create -y -n "${CONDA_ENV}" python=3.10 pip
  conda install -y -n "${CONDA_ENV}" -c conda-forge ffmpeg || true
fi
# shellcheck disable=SC1091
set +u
conda activate "${CONDA_ENV}"
set -u

export PYTHONNOUSERSITE=1
python -m pip install -U pip setuptools wheel packaging ninja
log "python=$(python --version) exe=$(which python)"

if python -c "import torch" 2>/dev/null; then
  log "torch already present: $(python -c 'import torch; print(torch.__version__, "cuda="+str(torch.version.cuda), "avail="+str(torch.cuda.is_available()))')"
else
  log "install torch 2.7.1+cu128 from download.pytorch.org"
  python -m pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
fi

python - <<'PY' | tee -a "$LOG_FILE"
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"gpu0={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
PY

log "install Motus python deps from Aliyun PyPI"
python -m pip install \
  "huggingface-hub>=0.34.0" \
  "hf_transfer" \
  "modelscope>=1.20.0" \
  "qwen-vl-utils>=0.0.11" \
  "tokenizers>=0.15.0" \
  "accelerate>=0.25.0" \
  "diffusers>=0.25.0" \
  "opencv-python>=4.8.0" \
  "Pillow>=10.0.0" \
  "imageio>=2.31.0" \
  "imageio-ffmpeg>=0.4.8" \
  "numpy>=1.24.0" \
  "h5py>=3.9.0" \
  "scipy>=1.11.0" \
  "pyyaml>=6.0" \
  "wandb>=0.16.0" \
  "tensorboard>=2.15.0" \
  "tqdm>=4.66.0" \
  "matplotlib>=3.7.0" \
  "seaborn>=0.12.0" \
  "omegaconf>=2.3.0" \
  "easydict>=1.13" \
  "ftfy>=6.3.1" \
  "safetensors" \
  "einops" \
  "sentencepiece" \
  "protobuf"

# Qwen3-VL 需要较新 transformers；官方 pin 5.0.0rc0，国内源没有则回退 4.57+
if ! python -c "import transformers; from transformers import Qwen3VLForConditionalGeneration" 2>/dev/null; then
  log "install transformers (Qwen3-VL)"
  python -m pip install "transformers==5.0.0rc0" \
    || python -m pip install "transformers>=4.57.0"
fi

python -m pip install "deepspeed>=0.18.3" || log "WARN deepspeed pip failed"
python -m pip install "decord>=0.6.0" || log "WARN decord skipped"

# flash-attn：Blackwell 需源码编 sm_120，失败不阻断（可走 SDPA）
if ! python -c "import flash_attn" 2>/dev/null; then
  CUDA_HOME_CANDIDATES=(
    "${CUDA_HOME:-}"
    "${HOME}/miniconda3/envs/lingbotva"
    "${HOME}/anaconda3/envs/lingbotvla"
    "${HOME}/miniconda3/envs/lingbotvla"
    /usr/local/cuda
  )
  for _cuda_home in "${CUDA_HOME_CANDIDATES[@]}"; do
    [[ -n "${_cuda_home}" && -x "${_cuda_home}/bin/nvcc" ]] || continue
    export CUDA_HOME="${_cuda_home}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    break
  done
  log "install flash-attn (CUDA_HOME=${CUDA_HOME:-unset} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST})"
  python -m pip install flash-attn --no-build-isolation \
    || log "WARN flash-attn skipped; Motus 可用 SDPA，训练会更慢"
fi

log "========== import check =========="
python - <<'PY' | tee -a "$LOG_FILE"
import importlib
ok = True
for m in ["torch", "torchvision", "transformers", "accelerate", "diffusers", "deepspeed", "omegaconf", "modelscope", "huggingface_hub"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"FAIL {m}: {e}")
        ok = False
try:
    from transformers import Qwen3VLForConditionalGeneration
    print("OK Qwen3VLForConditionalGeneration")
except Exception as e:
    print(f"FAIL Qwen3VL: {e}")
    ok = False
try:
    import flash_attn
    print(f"OK flash_attn {getattr(flash_attn, '__version__', '')}")
except Exception as e:
    print(f"WARN flash_attn: {e}")
print("gpu", __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else None)
if not ok:
    raise SystemExit(1)
PY

cat > "${MOTUS_ROOT}/env.sh" <<EOF
# source ${MOTUS_ROOT}/env.sh
unalias -a 2>/dev/null || true
if [[ -f "\${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "\${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "\${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
set +u
conda activate ${CONDA_ENV}
set +u
export MOTUS_ROOT="${MOTUS_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT}"
export HF_HUB_DISABLE_XET=1
export PIP_INDEX_URL="${PIP_INDEX_URL}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}"
export MODELSCOPE_CACHE="\${MODELSCOPE_CACHE:-${MOTUS_ROOT}/.modelscope}"
export HF_HOME="\${HF_HOME:-${MOTUS_ROOT}/.hf}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
export PYTHONPATH="${MOTUS_ROOT}:\${PYTHONPATH:-}"
cd "\${MOTUS_ROOT}"
echo "python=\$(which python) \$(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
EOF

cp -a "${MOTUS_ROOT}/env.sh" "${HERE}/env.sh"
log "wrote ${MOTUS_ROOT}/env.sh"
log "========== setup done =========="
