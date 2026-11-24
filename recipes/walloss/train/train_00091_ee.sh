#!/bin/bash
# 4 卡 WALL-OSS-0.5 SFT：tongbot 00091 相对 EE。
# jiangsuan-4: ROOT=/nvme2n1/wall-oss NPROC=4
set -euo pipefail
ROOT="${ROOT:-/nvme2n1/wall-oss}"
CFG="${CFG:-${ROOT}/datasets/00091_rm75_ee_v30/sft.yml}"
if [[ ! -f "${CFG}" ]]; then
  CFG="${ROOT}/datasets/00091_rm75_ee/sft.yml"
fi
NPROC="${NPROC:-4}"
PORT="${PORT:-29618}"
LOGDIR="${LOGDIR:-${ROOT}/outputs/walloss05_tongbot_ee_00091}"

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
export WALLX_SKIP_CUDA_EXT=1
export WALLX_DIST_TIMEOUT_MINUTES="${WALLX_DIST_TIMEOUT_MINUTES:-60}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
mkdir -p "${LOGDIR}"

echo "[train-ee-00091] $(date -Is) nproc=${NPROC} cfg=${CFG} python=$(which python)" | tee -a "${LOGDIR}/launch.log"
cd "${ROOT}/src/wall-x"
nohup python -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${PORT}" \
  "${ROOT}/code/train/run_sft.py" \
  --config "${CFG}" \
  --log_to_file \
  >> "${LOGDIR}/nohup.out" 2>&1 &
echo $! > "${LOGDIR}/train.pid"
echo "[train-ee-00091] pid=$(cat "${LOGDIR}/train.pid") log=${LOGDIR}/nohup.out"
# sidecar: GPU/PPU/NPU util+mem → ${LOGDIR}/hw.jsonl（不碰训练；HW_MONITOR=0 关）
# shellcheck disable=SC1091
for _hw_src in \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_hw_monitor.sh" \
  /home/xk/data_Driven/scripts/start_hw_monitor.sh \
  /nvme2n1/wall-oss/code/train/start_hw_monitor.sh; do
  [ -f "$_hw_src" ] || continue
  . "$_hw_src"
  break
done
unset _hw_src
if declare -F start_hw_monitor >/dev/null; then
  start_hw_monitor "${LOGDIR}/nohup.out" "${CUDA_VISIBLE_DEVICES:-}" "$(cat "${LOGDIR}/train.pid")" || true
fi
