#!/bin/bash
# π0.5 8-NPU DDP on Atlas A2（910B ×8）—— 00108 RGB+DEPTH HEAD AH=50 fixbase
# 投递：docker exec -d -w /home/pi05/openpi_src pi05 bash start_train_npu8_a2_00108_rgb_depth_head.sh
set -euo pipefail

cd /home/pi05/openpi_src

DS=/home/pi05/00108_20260922_RM_01_YJ_v21
YAML=configs/train_pi05_npu8_a2_00108_rgb_depth_head.yaml
if [ ! -f "$DS/meta/info.json" ]; then
  echo "refuse: dataset missing $DS"
  exit 1
fi
if [ ! -s "$DS/norm_stats.json" ]; then
  echo "refuse: $DS/norm_stats.json missing; compute_norm_stats first"
  exit 1
fi
if pgrep -f 'train_pi05_npu8_a2_00108_rgb_depth_head.yaml' >/dev/null; then
  echo "refuse: 00108 rgb_depth_head a2npu8 already up."
  exit 1
fi

export PYTHONPATH="/home/pi05/openpi_src/src:/home/pi05/openpi_src/packages/openpi-client/src:${PYTHONPATH:-}"
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
LOG_FILE="${CKPT_ROOT}/train_00108_a2_8npu_rgb_depth_head.log"

echo "=== π0.5 8x Ascend 910B (DDP) —— 00108 RGB+DEPTH HEAD AH=50 fixbase ==="
echo "Dataset: $DS"
echo "Config:  $YAML (bs=128, 16/NPU, lr 7e-5)"
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

exec taskset -c 0-319 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_from_yaml.py --config "$YAML" \
  > "$LOG_FILE" 2>&1
