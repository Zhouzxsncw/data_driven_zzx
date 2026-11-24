#!/usr/bin/env bash
# Motus 单机训练入口（默认 configs/local.yaml，DeepSpeed ZeRO-1）。
# 官方配方 8×80GB bs=8；96GB 单卡请 NPROC=1 BATCH 自改 yaml。
set -euo pipefail

MOTUS_ROOT="${MOTUS_ROOT:-/home/xk/VLA/Motus}"
NPROC="${NPROC:-1}"
CONFIG_FILE="${CONFIG_FILE:-configs/local.yaml}"
TASK="${TASK:-local}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ ! -f "${MOTUS_ROOT}/env.sh" ]]; then
  echo "missing ${MOTUS_ROOT}/env.sh ，先跑 setup_env.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${MOTUS_ROOT}/env.sh"

if [[ ! -f "${MOTUS_ROOT}/${CONFIG_FILE}" ]]; then
  echo "missing ${MOTUS_ROOT}/${CONFIG_FILE} ，先跑 link_pretrained.sh" >&2
  exit 1
fi

cd "${MOTUS_ROOT}"
export WANDB_MODE=offline
export OUTPUT_DIR="${OUTPUT_DIR:-${MOTUS_ROOT}/outputs/motus-${TASK}}"
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
