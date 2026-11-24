#!/bin/bash
# A3 新实验不要用这份。默认改走 recipes/pi05/train/huawei_npu16/（16 卡，bs256，lr 1e-4）。
# 这份只续已经在跑的 8 卡 exp_name。
# π0.5 8-NPU DDP on Atlas 800I A3 —— 槽 B（NPU 8-15），00036 AH=100
# 投递：docker exec -d -w /home/pi05/openpi_src pi05 bash start_train_npu8_00036_ah100.sh
# 槽 A：start_train_npu8_00036_ah15.sh（NPU 0-7 / HCCL 61000 / taskset 0-319）
# 镜像 torch 是 cpu 包；必须有 src/sitecustomize.py（transfer_to_npu）。宿主 nohup 没有 torchrun。
set -euo pipefail

cd /home/pi05/openpi_src

DS=/home/pi05/00036_20260804_RM_01_yj_v21
if [ ! -f "$DS/meta/info.json" ]; then
  echo "refuse: dataset missing $DS"
  exit 1
fi
if [ ! -s "$DS/norm_stats.json" ]; then
  echo "refuse: $DS/norm_stats.json missing; compute_norm_stats first"
  exit 1
fi
if pgrep -f 'train_pi05_npu8_00036_ah100.yaml' >/dev/null; then
  echo "refuse: 00036 AH=100 npu8 already up."
  exit 1
fi

export PYTHONPATH="/home/pi05/openpi_src/src:/home/pi05/openpi_src/packages/openpi-client/src:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export ASCEND_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
export HCCL_IF_BASE_PORT=62000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export PALIGEMMA_TOKENIZER_PATH=/home/pi05/models/tokenizer.model
export TORCHDYNAMO_DISABLE=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1
export TORCH_COMPILE_DISABLE=1

LOG_FILE="/home/pi05/openpi_src/checkpoints/train_00036_a3_8npu_ah100.log"

echo "=== π0.5 8x Ascend 910 (DDP, NPU 8-15) —— 00036 AH=100 ==="
echo "Dataset: $DS"
echo "Config:  configs/train_pi05_npu8_00036_ah100.yaml (bs=128, 16/NPU, lr 7e-5, AH=100)"
echo "Log:     $LOG_FILE"

[ -s "$LOG_FILE" ] && cp -f "$LOG_FILE" "$LOG_FILE.prev" || true

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

exec taskset -c 320-639 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_from_yaml.py --config configs/train_pi05_npu8_00036_ah100.yaml \
  > "$LOG_FILE" 2>&1
