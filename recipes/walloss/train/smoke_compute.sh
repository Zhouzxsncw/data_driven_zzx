#!/bin/bash
# 16 卡合成数据 compute smoke。容器内执行。
# 不抢别的训练容器；当前 16 张卡应空闲。
set -euo pipefail
ROOT="${ROOT:-/workspace}"
CKPT="${CKPT:-${ROOT}/pretrained/wall-oss-0.5}"
NPROC="${NPROC:-16}"
PORT="${PORT:-29616}"
LOG="${LOG:-${ROOT}/logs/smoke_compute.log}"

set +u
source /usr/local/PPU_SDK/envsetup.sh
# shellcheck disable=SC1091
source "${ROOT}/venv/bin/activate"
set -u

export CHECKPOINT_OPTIMIZATION="${CHECKPOINT_OPTIMIZATION:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export SMOKE_ATTN="${SMOKE_ATTN:-sdpa}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p "$(dirname "${LOG}")"

echo "[smoke] $(date -Is) nproc=${NPROC} ckpt=${CKPT} python=$(which python)" | tee "${LOG}"
cd /workspace/src/wall-x
python -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${PORT}" \
  /workspace/code/train/smoke_compute.py \
  --ckpt "${CKPT}" \
  "$@" \
  2>&1 | tee -a "${LOG}"
