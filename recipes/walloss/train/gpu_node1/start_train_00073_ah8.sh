#!/bin/bash
# gpu-node1 单卡 WALL-OSS-0.5 —— 00073 相对 EE，CHUNK SIZE / AH=8
# 在节点上执行（由 gpu-master scp 后 setsid nohup）。
set -euo pipefail

source /data/wall-oss/env.sh
set -euo pipefail

export LD_LIBRARY_PATH=/data/wall-oss/conda/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CHECKPOINT_OPTIMIZATION="${CHECKPOINT_OPTIMIZATION:-true}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES=0

CFG=/data/wall-oss/code/train/sft_tongbot_ee_00073_ah8.yml
LOGDIR=/data/wall-oss/outputs/walloss05_tongbot_ee_00073_ah8
LOG=/data/wall-oss/logs/train_00073_ah8.log
PORT="${PORT:-29628}"

if pgrep -f 'sft_tongbot_ee_00073_ah8.yml' >/dev/null; then
  echo "refuse: 00073 AH=8 already up."
  exit 1
fi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${USED:-99999}" -gt 8000 ]; then
  echo "refuse: GPU still occupied (${USED} MiB). stop the other job first."
  exit 1
fi

mkdir -p "$LOGDIR" /data/wall-oss/logs
cp -f "$CFG" "$LOGDIR/sft.yml"
[ -s "$LOG" ] && cp -f "$LOG" "$LOG.prev"

echo "=== WALL-OSS-0.5 1× RTX PRO 6000 —— 00073 EE AH=8 ==="
echo "Dataset: /data/wall-oss/datasets/00073_rm75_ee_v30 (194273 frames)"
echo "Config:  $CFG (bs=16 accum=4 global=64, lr 5e-5, AH=8, 10 epoch)"
echo "Log:     $LOG"
echo "[00073-ah8] $(date -Is) nproc=1 cfg=${CFG} python=$(which python)" | tee -a "${LOGDIR}/launch.log"

cd /data/wall-oss/code/train
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

exec /data/wall-oss/conda/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  --master_port="${PORT}" \
  run_sft.py --config "$CFG" --log_to_file \
  > "$LOG" 2>&1
