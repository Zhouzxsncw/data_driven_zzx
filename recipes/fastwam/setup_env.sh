#!/usr/bin/env bash
# FastWAM 训练环境：gpu-master / gpu-node1（Blackwell sm_120）
# 国内源：阿里云 PyPI、ModelScope（Wan 权重）、HF 镜像兜底。
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET=1
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export DS_BUILD_AIO=0
export DS_BUILD_SPARSE_ATTN=0

CONDA_ENV="${CONDA_ENV:-fastwam}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="${FASTWAM_ROOT:-/home/xk/VLA/FastWAM}"
LOG_FILE="${LOG_FILE:-${FASTWAM_ROOT}/setup_env.log}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

mkdir -p "$(dirname "$LOG_FILE")" "$FASTWAM_ROOT"

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

log "========== FastWAM setup =========="
log "FASTWAM_ROOT=$FASTWAM_ROOT CONDA_ENV=$CONDA_ENV PIP_INDEX_URL=$PIP_INDEX_URL"
log "DIFFSYNTH_DOWNLOAD_SOURCE=$DIFFSYNTH_DOWNLOAD_SOURCE HF_ENDPOINT=$HF_ENDPOINT"

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

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log "torch already ok: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
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

log "install python deps from Aliyun PyPI"
python -m pip install -e "${FASTWAM_ROOT}" --no-deps
python -m pip install \
  "accelerate==1.12.0" \
  "av==15.1.0" \
  "deepspeed==0.18.7" \
  "einops==0.8.1" \
  "huggingface-hub==0.29.2" \
  "hydra-core==1.3.2" \
  "imageio==2.37.0" \
  "imageio-ffmpeg==0.6.0" \
  "jsonlines==4.0.0" \
  "modelscope==1.34.0" \
  "numpy==2.2.6" \
  "omegaconf==2.3.0" \
  "pandas==2.2.3" \
  "pillow==12.0.0" \
  "pyarrow==24.0.0" \
  "regex" \
  "rich" \
  "safetensors==0.5.3" \
  "termcolor==2.5.0" \
  "tqdm==4.68.3" \
  "transformers==4.49.0" \
  "wandb==0.23.1" \
  "datasets==4.8.5" \
  "gitpython" \
  "boto3" \
  "packaging==25.0"

python -m pip install torchcodec==0.4.0 || log "WARN torchcodec skipped"

log "========== import check =========="
python - <<'PY' | tee -a "$LOG_FILE"
import importlib
for m in ["torch", "torchvision", "accelerate", "deepspeed", "hydra", "omegaconf", "modelscope", "fastwam"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"FAIL {m}: {e}")
        raise
print("gpu", __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else None)
PY

cat > "${FASTWAM_ROOT}/env.sh" <<EOF
# source ${FASTWAM_ROOT}/env.sh
unalias -a 2>/dev/null || true
if [[ -f "\${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "\${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "\${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
set +u
conda activate ${CONDA_ENV}
set +u
export FASTWAM_ROOT="${FASTWAM_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT}"
export HF_HUB_DISABLE_XET=1
export PIP_INDEX_URL="${PIP_INDEX_URL}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ROOT}/checkpoints"
export MODELSCOPE_CACHE="\${MODELSCOPE_CACHE:-${FASTWAM_ROOT}/.modelscope}"
export HF_HOME="\${HF_HOME:-${FASTWAM_ROOT}/.hf}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
cd "\${FASTWAM_ROOT}"
echo "python=\$(which python) \$(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
EOF

# recipes 侧也放一份，方便 data_Driven 启动脚本 source
cp -a "${FASTWAM_ROOT}/env.sh" "${HERE}/env.sh"
log "wrote ${FASTWAM_ROOT}/env.sh"
log "========== setup done =========="
