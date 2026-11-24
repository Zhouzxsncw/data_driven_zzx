#!/bin/bash
# A2 一路 8 卡：00108 fps5，每卡 24，global 192。
set -euo pipefail

cd /home/pi05/openpi_src

if pgrep -f 'scripts/train_from_yaml.py' >/dev/null; then
  echo "refuse: A2 8-card job needs all NPUs; train_from_yaml already running."
  exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
export HCCL_IF_BASE_PORT=61000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export PALIGEMMA_TOKENIZER_PATH=/home/pi05/models/tokenizer.model
export TORCHDYNAMO_DISABLE=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1
export TORCH_COMPILE_DISABLE=1

CKPT_ROOT=/home/software/pi05/checkpoints
if [ ! -d "$CKPT_ROOT" ]; then
  echo "refuse: $CKPT_ROOT not visible; mount /home/software into the pi05 container first"
  exit 1
fi
LOG_FILE="${CKPT_ROOT}/train_00108_a2_8npu_fps5.log"
mkdir -p "${CKPT_ROOT}/00108_PI05_a2npu8_bs192_lr8.57e-5_ah50_fps5_fixbase/resource_monitor"

echo "=== π0.5 8x Ascend 910B —— 00108 fps5 ==="
echo "Config: configs/train_pi05_npu8_a2_00108_fps5.yaml (bs=192, 24/NPU, lr 8.57e-5)"
echo "Log:    $LOG_FILE"

[ -s "$LOG_FILE" ] && cp -f "$LOG_FILE" "$LOG_FILE.prev"

# npu-smi 只在宿主。容器里起 sidecar 会写成 backend=none。宿主另挂。
export HW_MONITOR=0

# shellcheck disable=SC1091
for _hw_src in \
  /home/pi05/openpi_src/start_hw_monitor.sh \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_hw_monitor.sh"; do
  [ -f "$_hw_src" ] || continue
  . "$_hw_src"
  break
done
unset _hw_src
if declare -F start_hw_monitor >/dev/null; then
  start_hw_monitor "${LOG:-${LOG_FILE}}" || true
fi

exec taskset -c 0-319 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_from_yaml.py --config configs/train_pi05_npu8_a2_00108_fps5.yaml \
  > "$LOG_FILE" 2>&1
