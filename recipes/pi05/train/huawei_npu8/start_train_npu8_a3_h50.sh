#!/bin/bash
# π0.5 8-NPU DDP on Atlas A3 —— NPU 0-7，mix 块长 50
# 投递：docker exec -d -w /home/pi05/openpi_src pi05 bash start_train_npu8_a3_h50.sh
set -euo pipefail

cd /home/pi05/openpi_src

DS=/home/pi05/mix_108_36_91_73
YAML=configs/train_pi05_a3_h50.yaml
if [ ! -f "$DS/meta/info.json" ]; then
  echo "refuse: dataset missing $DS"
  exit 1
fi
if [ ! -s "$DS/norm_stats.json" ]; then
  echo "refuse: $DS/norm_stats.json missing"
  exit 1
fi
if [ ! -s "$DS/meta/episode_split.json" ]; then
  echo "refuse: $DS/meta/episode_split.json missing"
  exit 1
fi
if pgrep -f 'train_pi05_a3_h50.yaml' >/dev/null; then
  echo "refuse: mix h50 already up."
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
export PI05_ACTION_DIM_GUARD=0
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1
export TORCH_COMPILE_DISABLE=1

CKPT_ROOT=/data/pi05/checkpoints
if ! findmnt -T "$CKPT_ROOT" | grep -q nvme1n1; then
  echo "refuse: $CKPT_ROOT is not on /data (nvme1)"
  exit 1
fi
LOG_FILE="${CKPT_ROOT}/train_mix_a3_h50.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "=== π0.5 8x Ascend 910 (DDP, NPU 0-7) —— mix h50 ==="
echo "Dataset: $DS"
echo "Config:  $YAML (bs=256, 32/NPU, lr 9.9e-5, horizon 50)"
echo "Log:     $LOG_FILE"

[ -s "$LOG_FILE" ] && cp -f "$LOG_FILE" "$LOG_FILE.prev" || true

# shellcheck disable=SC1091
for _hw_src in \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_hw_monitor.sh" \
  /home/pi05/openpi_src/start_hw_monitor.sh; do
  [ -f "$_hw_src" ] || continue
  . "$_hw_src"
  break
done
unset _hw_src
if declare -F start_hw_monitor >/dev/null; then
  start_hw_monitor "${LOG:-${LOG_FILE:-${LOGDIR:+$LOGDIR/nohup.out}}}" || true
fi

exec taskset -c 0-319 torchrun --standalone --nnodes=1 --nproc_per_node=8 --master_port=29501 \
  scripts/train_from_yaml.py --config "$YAML" \
  > "$LOG_FILE" 2>&1
