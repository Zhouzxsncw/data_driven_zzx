#!/bin/bash
# π0.5 4-NPU DDP —— NPU 0-3，数据集 00036，推荐档 bs64 / lr 3.5e-5 / AH16
# 00073 8 卡仍占 8-15 / 核 320-639；本 run 只绑 0-159，HCCL 61000。
set -euo pipefail

cd /home/pi05/openpi_src

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_IF_BASE_PORT=61000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export PALIGEMMA_TOKENIZER_PATH=/home/pi05/models/tokenizer.model
export TORCHDYNAMO_DISABLE=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1
export TORCH_COMPILE_DISABLE=1

LOG_FILE="/home/pi05/openpi_src/checkpoints/train_00036_4npu_ah16.log"

echo "=== π0.5 training: 4x Ascend 910 (DDP, NPU 0-3) —— 00036 npu4 recommend ==="
echo "Dataset: /home/pi05/00036_20260804_RM_01_yj_v21 (50440 frames)"
echo "Config:  configs/train_pi05_npu4_00036.yaml (bs=64 total, 16/NPU, lr 3.5e-5, AH=16)"
echo "Log:     $LOG_FILE"

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

exec taskset -c 0-159 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_from_yaml.py --config configs/train_pi05_npu4_00036.yaml \
  > "$LOG_FILE" 2>&1
