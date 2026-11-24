# source /home/xk/VLA/FastWAM/env.sh
unalias -a 2>/dev/null || true
# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
set +u
conda activate fastwam
set +u
export PATH="${HOME}/miniconda3/envs/fastwam/bin:${PATH}"
hash -r 2>/dev/null || true
export FASTWAM_ROOT="/home/xk/VLA/FastWAM"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export PIP_TRUSTED_HOST="mirrors.aliyun.com"
export DIFFSYNTH_DOWNLOAD_SOURCE="modelscope"
export DIFFSYNTH_MODEL_BASE_PATH="/home/xk/VLA/FastWAM/checkpoints"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/home/xk/VLA/FastWAM/.modelscope}"
export HF_HOME="${HF_HOME:-/home/xk/VLA/FastWAM/.hf}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME="${CUDA_HOME:-/home/xk/miniconda3/envs/lingbotva}"
export PATH="${CUDA_HOME}/bin:${PATH}"
cd "${FASTWAM_ROOT}"
echo "python=$(which python) $(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
