#!/bin/bash
# π0.5 4-NPU —— NPU 8-11，00036，AH=75
# 与 AH16@0-3、AH25@4-7、AH100@12-15 并行。
set -euo pipefail

cd /home/pi05/openpi_src

if pgrep -f 'train_pi05_npu4_00036_ah75.yaml' >/dev/null; then
  echo "refuse: 00036 AH=75 already up."
  exit 1
fi
if pgrep -f 'train_pi05_npu4_00073' >/dev/null; then
  echo "refuse: 00073 npu4 still occupying NPU 4-15."
  exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11
export HCCL_IF_BASE_PORT=64000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export PALIGEMMA_TOKENIZER_PATH=/home/pi05/models/tokenizer.model
export TORCHDYNAMO_DISABLE=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1
export TORCH_COMPILE_DISABLE=1

LOG_FILE="/home/pi05/openpi_src/checkpoints/train_00036_4npu_ah75.log"

echo "=== π0.5 4x Ascend 910 (NPU 8-11) —— 00036 AH=75 ==="
echo "Config: configs/train_pi05_npu4_00036_ah75.yaml (bs=64, lr 3.5e-5, AH=75)"
echo "Log:    $LOG_FILE"

[ -s "$LOG_FILE" ] && cp -f "$LOG_FILE" "$LOG_FILE.prev"

# sidecar: GPU/PPU/NPU util+mem → *.hw.jsonl（不碰训练；HW_MONITOR=0 关）
# shellcheck disable=SC1091
for _hw_src in \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_hw_monitor.sh" \
  /home/xk/data_Driven/scripts/start_hw_monitor.sh \
  /home/pi05/openpi_src/start_hw_monitor.sh \
  /workspace/openpi/cigai_train/openpi-jiangsuan/scripts/start_hw_monitor.sh \
  /workspace/code/train/start_hw_monitor.sh \
  /nvme2n1/wall-oss/code/train/start_hw_monitor.sh; do
  [ -f "$_hw_src" ] || continue
  . "$_hw_src"
  break
done
unset _hw_src
if declare -F start_hw_monitor >/dev/null; then
  start_hw_monitor "${LOG:-${LOG_FILE:-${LOGDIR:+$LOGDIR/nohup.out}}}" || true
fi

exec taskset -c 320-479 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_from_yaml.py --config configs/train_pi05_npu4_00036_ah75.yaml \
  > "$LOG_FILE" 2>&1
