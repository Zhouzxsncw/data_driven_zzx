#!/usr/bin/env bash
# h100-103: Motus 8×H100 训练入口（ZeRO-1，NPROC 控制卡数）
# 用法: NPROC=8 CONFIG_FILE=configs/tongbot_00091.yaml TASK=tongbot_00091 \
#       bash /data1/motus/recipes/launch_train_h100.sh
set -euo pipefail

MOTUS_ROOT="${MOTUS_ROOT:-/data1/motus/Motus}"
NPROC="${NPROC:-8}"
CONFIG_FILE="${CONFIG_FILE:-configs/tongbot_00091.yaml}"
TASK="${TASK:-tongbot_00091}"
MASTER_PORT="${MASTER_PORT:-29501}"
VENV="${VENV:-/data1/motus-env}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "missing ${VENV}，先跑 setup_env_h100.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ ! -f "${MOTUS_ROOT}/${CONFIG_FILE}" ]]; then
  echo "missing ${MOTUS_ROOT}/${CONFIG_FILE}" >&2
  exit 1
fi

cd "${MOTUS_ROOT}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OUTPUT_DIR="${OUTPUT_DIR:-${MOTUS_ROOT}/outputs/${TASK}}"
mkdir -p "${OUTPUT_DIR}"

echo "[train] nproc=${NPROC} config=${CONFIG_FILE} output=${OUTPUT_DIR}"
torchrun \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  train/train.py \
  --deepspeed configs/zero1.json \
  --config "${CONFIG_FILE}" \
  --run_name "${TASK}" \
  --report_to tensorboard
