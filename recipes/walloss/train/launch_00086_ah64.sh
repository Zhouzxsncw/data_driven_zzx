#!/bin/bash
# 本机单卡 WALL-OSS-0.5 SFT：00086 相对 EE，chunk size / action_horizon = 64
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CFG="${CFG:-${REPO}/recipes/walloss/train/sft_tongbot_ee_00086_ah64.yml}"
LOGDIR="${LOGDIR:-/data/wall-oss/outputs/walloss05_tongbot_ee_00086_ah64}"
PORT="${PORT:-29618}"
ENTRY="${ENTRY:-${REPO}/recipes/walloss/train/run_sft.py}"

# shellcheck disable=SC1091
source /data/wall-oss/env.sh
set -euo pipefail

mkdir -p "${LOGDIR}"
cp -f "${CFG}" "${LOGDIR}/sft.yml"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CHECKPOINT_OPTIMIZATION="${CHECKPOINT_OPTIMIZATION:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[00086-ah64] $(date -Is) nproc=1 cfg=${CFG} python=$(which python)" | tee -a "${LOGDIR}/launch.log"
cd /data/wall-oss/src/wall-x
# sidecar: GPU/PPU/NPU util+mem → *.hw.jsonl（不碰训练；HW_MONITOR=0 关）
# shellcheck disable=SC1091
for _hw_src in \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_hw_monitor.sh" \
  /home/xk/data_Driven/scripts/start_hw_monitor.sh; do
  [ -f "$_hw_src" ] || continue
  . "$_hw_src"
  break
done
unset _hw_src
if declare -F start_hw_monitor >/dev/null; then
  start_hw_monitor "${LOGDIR}/sft.log" || true
fi
exec python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  --master_port="${PORT}" \
  "${ENTRY}" \
  --config "${CFG}" \
  --log_to_file
