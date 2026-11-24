#!/usr/bin/env bash
# 江算 16/16b/16c · 8 卡一路。在 openpi_train 容器里跑：
#   YAML=configs/train_pi05_ppu8_00036.yaml \
#   GPUS=0,1,2,3,4,5,6,7 PORT=29701 LOG=/workspace/openpi/train_00036_ppu8.log \
#   bash scripts/start_train_ppu8.sh
# 槽 B：GPUS=8,9,10,11,12,13,14,15 PORT=29798
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/openpi/cigai_train/openpi-jiangsuan}"
YAML="${YAML:-configs/train_pi05_ppu8_00036.yaml}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-29701}"
NPROC="${NPROC:-8}"
LOG="${LOG:-/workspace/openpi/train_ppu8.log}"

cd "${REPO_DIR}"

if pgrep -af "train_from_yaml.py --config ${YAML}" | grep -v pgrep >/dev/null; then
  echo "[skip] 已在跑："
  pgrep -af "train_from_yaml.py --config ${YAML}" | grep -v pgrep || true
  exit 0
fi

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/packages/openpi-client/src:${REPO_DIR}/cigia_lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/workspace/openpi/models/pi05_fixed/tokenizer.model}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
# 动作维不在这里写死。训练循环先读数据集 meta/info.json，再按那一维监控。

echo "[launch] gpus=${GPUS} nproc=${NPROC} port=${PORT} ${YAML} -> ${LOG}"

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

exec python3 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${NPROC}" --master_port="${PORT}" \
  scripts/train_from_yaml.py --config "${YAML}" \
  >> "${LOG}" 2>&1
