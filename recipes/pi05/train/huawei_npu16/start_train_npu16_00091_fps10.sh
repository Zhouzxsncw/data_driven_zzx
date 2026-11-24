#!/bin/bash
# A3 一路 16 卡：00091 fps10，每卡 24，global 384。
set -euo pipefail

cd /home/pi05/openpi_src

if pgrep -f 'scripts/train_from_yaml.py' >/dev/null; then
  echo "refuse: A3 16-card job needs all NPUs; train_from_yaml already running."
  exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
export ASCEND_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-61000}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export PALIGEMMA_TOKENIZER_PATH=/home/pi05/models/tokenizer.model
export TORCHDYNAMO_DISABLE=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1
export TORCH_COMPILE_DISABLE=1

LOG_FILE="/home/pi05/openpi_src/checkpoints/train_00091_16npu_fps10.log"
mkdir -p "/home/pi05/openpi_src/checkpoints/00091_PI05_npu16_bs384_lr1.225e-4_ah50_fps10_fixbase/resource_monitor"

echo "=== π0.5 training: 16x Ascend 910 —— 00091 fps10 fixbase ==="
echo "Config:  configs/train_pi05_npu16_00091_fps10.yaml (bs=384 total, 24/NPU, lr 1.225e-4)"
echo "Log:     $LOG_FILE"

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

exec taskset -c "${CPUS:-0-639}" torchrun --standalone --nnodes=1 --nproc_per_node=16 \
  scripts/train_from_yaml.py --config configs/train_pi05_npu16_00091_fps10.yaml \
  > "$LOG_FILE" 2>&1
