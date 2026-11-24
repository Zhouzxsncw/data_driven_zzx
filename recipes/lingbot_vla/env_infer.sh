# LingBot-VLA V2 服务端环境（conda lingbotvla）
# source recipes/lingbot_vla/env_infer.sh
unalias -a 2>/dev/null || true
# shellcheck disable=SC1091
source /home/xk/miniconda3/etc/profile.d/conda.sh
set +u
conda activate lingbotvla
set +u

export LINGBOT_VLA_REPO="${LINGBOT_VLA_REPO:-/home/xk/VLA/lingbot-vla-v2}"
export QWEN3VL_PATH="${QWEN3VL_PATH:-/home/xk/VLA/lingbot-vla-v2/pretrained/qwen3-tokenizer}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

echo "python=$(which python) $(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
