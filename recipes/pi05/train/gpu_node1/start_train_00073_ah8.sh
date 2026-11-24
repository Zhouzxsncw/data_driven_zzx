#!/bin/bash
# gpu-node1 单卡 JAX π0.5 —— 00073，CHUNK SIZE / AH=8
# 在节点上执行（由 gpu-master scp 后 setsid nohup）。
set -euo pipefail

REPO=/home/cqcigai/cigai-agilex-train-openpi
YAML=$REPO/configs/train_pi05_00073_ah8.yaml
LOG=/data/openpi/train_00073_ah8.log

if pgrep -f 'train_pi05_00073_ah8.yaml' >/dev/null; then
  echo "refuse: 00073 AH=8 already up."
  exit 1
fi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${USED:-99999}" -gt 8000 ]; then
  echo "refuse: GPU still occupied (${USED} MiB). stop the other job first."
  exit 1
fi

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

mkdir -p /data/openpi
[ -s "$LOG" ] && cp -f "$LOG" "$LOG.prev"

echo "=== π0.5 JAX 1× RTX PRO 6000 —— 00073 AH=8 ==="
echo "Dataset: /home/cqcigai/datasets/00073_20260818_RM_01_yj_21 (194273 frames)"
echo "Config:  $YAML (bs=32, lr 2.5e-5, AH=8, 25 epoch)"
echo "Log:     $LOG"

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

exec "$REPO/.venv/bin/python" scripts/train_from_yaml.py --config "$YAML" \
  > "$LOG" 2>&1
