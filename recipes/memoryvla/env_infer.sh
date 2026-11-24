# source /home/xk/VLA/memoryvla_staging/env_infer.sh
# MemoryVLA 服务端环境（conda memvla）
unalias -a 2>/dev/null || true
# shellcheck disable=SC1091
source /home/xk/miniconda3/etc/profile.d/conda.sh
conda activate memvla
# shellcheck disable=SC1091
source /home/xk/VLA/MemoryVLA/env.sh

export MEMVLA_ROOT="${MEMVLA_ROOT:-/home/xk/VLA/MemoryVLA}"
export MEMVLA_CKPT="${MEMVLA_CKPT:-/data/memoryvla/outputs/00086/memvla_00086_80k--image_aug/checkpoints/step-010000-epoch-00-loss=0.0135.pt}"
export PYTHONPATH="${MEMVLA_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

echo "python=$(which python) $(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
