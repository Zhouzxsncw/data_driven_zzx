#!/bin/bash
# DISABLED 2026-09-15：00086 传完也不起训；NPU 8-11 改给 00073 AH=125。
echo "refuse: 00086 training disabled (NPU 8-11 used by 00073 AH=125)."
exit 1

cd /home/pi05/openpi_src

DS=/home/pi05/00086_20260827_RM_01_YJ_ZL_21
if [ ! -d "$DS" ] || [ ! -f "$DS/meta/info.json" ]; then
  echo "refuse: dataset missing $DS"
  exit 1
fi
if pgrep -f 'train_pi05_npu4_00086' >/dev/null; then
  echo "refuse: 00086 npu4 already up."
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

LOG_FILE="/home/pi05/openpi_src/checkpoints/train_00086_4npu_ah15.log"

echo "=== π0.5 4x Ascend 910 (NPU 8-11) —— 00086 AH=15 ==="
echo "Config: configs/train_pi05_npu4_00086_ah15.yaml (bs=64, lr 3.5e-5, AH=15)"
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
  scripts/train_from_yaml.py --config configs/train_pi05_npu4_00086_ah15.yaml \
  > "$LOG_FILE" 2>&1
