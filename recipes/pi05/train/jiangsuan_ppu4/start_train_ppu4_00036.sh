#!/usr/bin/env bash
# 江算 4 卡 π0.5：js1 同款 openpi-jiangsuan + lerobot venv，torchrun 4 进程。
# 在 jiangsuan-4 上执行：
#   bash /home/ubuntu/cigai_train/openpi-jiangsuan/scripts/start_train_ppu4_00036.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/cigai_train/openpi-jiangsuan}"
YAML="${YAML:-configs/train_pi05_ppu4_00036.yaml}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-29701}"
LOG_DIR="${LOG_DIR:-/nvme2n1/outputs}"
LOG="${LOG:-${LOG_DIR}/train_00036_ppu4_bs384_ah50.log}"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

if pgrep -af "train_from_yaml.py --config ${YAML}" | grep -v pgrep >/dev/null; then
  echo "[skip] 已在跑："
  pgrep -af "train_from_yaml.py --config ${YAML}" | grep -v pgrep || true
  exit 0
fi

# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/env_ppu.sh"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

echo "[launch] gpus=${GPUS} port=${PORT} ${YAML} -> ${LOG}"
echo "[ckpt]   ${REPO_DIR} + checkpoint_base_dir=/nvme2n1/outputs"

[ -s "${LOG}" ] && cp -f "${LOG}" "${LOG}.prev"

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

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 --master_port="${PORT}" \
  scripts/train_from_yaml.py --config "${YAML}" \
  > "${LOG}" 2>&1
