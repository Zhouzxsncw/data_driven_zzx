# LingBot-VA 服务端环境（conda lingbotva）
# source recipes/lingbot_va/env_infer.sh
unalias -a 2>/dev/null || true
# shellcheck disable=SC1091
source /home/xk/miniconda3/etc/profile.d/conda.sh
set +u
conda activate lingbotva
set +u

export LINGBOT_VA_CODE="${LINGBOT_VA_CODE:-/home/xk/VLA/lingbot-va}"
export LINGBOT_VA_RESOLVED_ROOT="${LINGBOT_VA_RESOLVED_ROOT:-/home/xk/VLA/lingbot-va-pretrained/_resolved_00036_va}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

echo "python=$(which python) $(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
