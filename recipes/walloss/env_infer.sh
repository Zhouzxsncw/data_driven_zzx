# source recipes/walloss/env_infer.sh
# WALL-OSS 服务端：本机 /data/wall-oss conda（torch 2.10+cu128, sm_120）
unalias -a 2>/dev/null || true
# shellcheck disable=SC1091
source /data/wall-oss/env.sh

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALLOSS_SCRIPTS="${WALLOSS_SCRIPTS:-${_HERE}}"
if [[ ! -f "${WALLOSS_SCRIPTS}/ee_infer_pipeline.py" ]]; then
  WALLOSS_SCRIPTS=/home/xk/VLA/wall_oss_ppu_setup/scripts
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export RM_SDK_ROOT="${RM_SDK_ROOT:-/home/xk/RM_API}"
export WALLX_CODE="${WALLX_CODE:-/data/wall-oss/src/wall-x}"
export WALLOSS_CKPT="${WALLOSS_CKPT:-/data/wall-oss/outputs/walloss05_tongbot_ee_00086/9}"
export PYTHONPATH="${WALLX_CODE}:${WALLOSS_SCRIPTS}:${RM_SDK_ROOT}:${PYTHONPATH:-}"
export WALLX_VISION_ATTN_IMPLEMENTATION="${WALLX_VISION_ATTN_IMPLEMENTATION:-flash_attention_2}"

echo "python=$(which python) $(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
