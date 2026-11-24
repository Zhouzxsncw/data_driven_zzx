# source /home/xk/VLA/Motus/env.sh
unalias -a 2>/dev/null || true
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
set +u
conda activate motus
set +u
export MOTUS_ROOT="/home/xk/VLA/Motus"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export PIP_TRUSTED_HOST="mirrors.aliyun.com"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/home/xk/VLA/Motus/.modelscope}"
export HF_HOME="${HF_HOME:-/home/xk/VLA/Motus/.hf}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="12.0"
export PYTHONPATH="/home/xk/VLA/Motus:${PYTHONPATH:-}"
cd "${MOTUS_ROOT}"
echo "python=$(which python) $(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
