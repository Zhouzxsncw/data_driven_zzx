#!/bin/bash
# 8 卡 WALL-OSS-0.5 SFT：tongbot 00086 相对 EE。16 卡机占一槽，不要 NPROC=16。
set -euo pipefail
ROOT="${ROOT:-/workspace}"
CFG="${CFG:-${ROOT}/code/train/sft_tongbot_ee_00086_8gpu_bs8.yml}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
PORT="${PORT:-29618}"
LOGDIR="${LOGDIR:-${ROOT}/outputs/walloss05_tongbot_ee_00086_8gpu}"

set +u
source /usr/local/PPU_SDK/envsetup.sh
# shellcheck disable=SC1091
source "${ROOT}/venv/bin/activate"
set -u

export CHECKPOINT_OPTIMIZATION="${CHECKPOINT_OPTIMIZATION:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export WALLX_DIST_TIMEOUT_MINUTES="${WALLX_DIST_TIMEOUT_MINUTES:-60}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
mkdir -p "${LOGDIR}"

echo "[train-ee-00086] $(date -Is) nproc=${NPROC} cfg=${CFG} python=$(which python)" | tee -a "${LOGDIR}/launch.log"
cd /workspace/src/wall-x
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

exec python -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${PORT}" \
  /workspace/code/train/run_sft.py \
  --config "${CFG}" \
  --log_to_file \
  >> "${LOGDIR}/nohup.out" 2>&1
