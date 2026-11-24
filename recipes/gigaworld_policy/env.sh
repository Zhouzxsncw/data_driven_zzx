# source /home/xk/VLA/giga-world-policy/env.sh
unalias -a 2>/dev/null || true
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
set +u
conda activate gigaworld-policy
set +u
export GWP_ROOT="/home/xk/VLA/giga-world-policy"
export PRETRAINED_ROOT="/data/giga-world-policy/pretrained"
export GWP05_TRANSFORMER_PRETRAINED="/data/giga-world-policy/pretrained/Giga-World-Policy-0.5"
export GWP_WAN_PRETRAINED="/data/giga-world-policy/pretrained/Wan2.2-TI2V-5B-Diffusers"
export BASE_MODEL="${GWP_WAN_PRETRAINED}"
export CHECKPOINT="${GWP05_TRANSFORMER_PRETRAINED}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export PIP_TRUSTED_HOST="mirrors.aliyun.com"
export HF_HOME="${HF_HOME:-/data/giga-world-policy/pretrained/.hf}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="12.0"
export GIGA_MODELS_LIGHT_IMPORT=1
export PYTHONPATH="/home/xk/VLA/giga-world-policy:/home/xk/VLA/giga-world-policy/third_party/giga-train:/home/xk/VLA/giga-world-policy/third_party/giga-models:/home/xk/VLA/giga-world-policy/third_party/giga-datasets:${PYTHONPATH:-}"
cd "${GWP_ROOT}"
echo "python=$(which python) $(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo "GWP05=${GWP05_TRANSFORMER_PRETRAINED}"
echo "WAN=${GWP_WAN_PRETRAINED}"
