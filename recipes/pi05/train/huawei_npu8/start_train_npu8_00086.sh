#!/bin/bash
# A3 新实验不要用这份。默认改走 recipes/pi05/train/huawei_npu16/（16 卡，bs256，lr 1e-4）。
# 这份只续已经在跑的 8 卡 exp_name。
# π0.5 8-NPU DDP on Atlas 800I A3 —— 00086。默认槽 A（NPU 0-7）。
# 槽 B：ASCEND_RT_VISIBLE_DEVICES=8-15、HCCL 62000、taskset 320-639。
# 0-7 已被 00036 npu8 占用时不要用这份，改槽 B 或等用户说停。
set -euo pipefail

cd /home/pi05/openpi_src

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-61000}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export PALIGEMMA_TOKENIZER_PATH=/home/pi05/models/tokenizer.model
export TORCHDYNAMO_DISABLE=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1
export TORCH_COMPILE_DISABLE=1

LOG_FILE="/home/pi05/openpi_src/checkpoints/train_00086_8npu_fixbase.log"
CPUS="${CPUS:-0-319}"

echo "=== π0.5 training: 8x Ascend 910 (DDP, NPU ${ASCEND_RT_VISIBLE_DEVICES}) —— 00086 fixbase ==="
echo "Config:  configs/train_pi05_npu8_00086.yaml (bs=128 total, 16/NPU, lr 7e-5)"
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

exec taskset -c "$CPUS" torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_from_yaml.py --config configs/train_pi05_npu8_00086.yaml \
  > "$LOG_FILE" 2>&1
