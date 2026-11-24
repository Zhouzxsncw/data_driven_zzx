#!/usr/bin/env bash
# 把 FastWAM 代码 + Wan2.2 权重推到 gpu-node1，并在远端装环境。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_CFG="${SSH_CFG:-/home/xk/VLA/ssh_config}"
HOST="${HOST:-gpu-node1}"
LOCAL_FASTWAM="${LOCAL_FASTWAM:-/home/xk/VLA/FastWAM}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/fastwam}"
REMOTE_FASTWAM="${REMOTE_FASTWAM:-${REMOTE_ROOT}/FastWAM}"
LOCAL_WAN22="${LOCAL_WAN22:-/home/xk/VLA/dreamzero_staging/pretrained/Wan2.2-TI2V-5B}"

ssh_cmd() {
  ssh -F "${SSH_CFG}" -o ConnectTimeout=20 -o BatchMode=yes "${HOST}" "$@"
}

echo "[1/5] mkdir ${HOST}:${REMOTE_ROOT}"
ssh_cmd "mkdir -p ${REMOTE_FASTWAM} ${REMOTE_ROOT}/runs ${REMOTE_ROOT}/logs ${REMOTE_FASTWAM}/checkpoints/Wan-AI"

echo "[2/5] rsync FastWAM code"
rsync -az --info=progress2 \
  -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' \
  --exclude '.hf' --exclude '.modelscope' --exclude 'runs' \
  --exclude 'evaluate_results' --exclude 'third_party' \
  "${LOCAL_FASTWAM}/" "${HOST}:${REMOTE_FASTWAM}/"

echo "[3/5] rsync recipes/fastwam scripts"
rsync -az --info=progress2 \
  -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  "${HERE}/" "${HOST}:${REMOTE_ROOT}/recipes/"

echo "[4/5] rsync Wan2.2-TI2V-5B (reuse local 32G, skip .cache)"
if [[ -d "${LOCAL_WAN22}" ]]; then
  rsync -a --info=progress2 \
    -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
    --exclude '.cache' \
    "${LOCAL_WAN22}/" "${HOST}:${REMOTE_FASTWAM}/checkpoints/Wan-AI/Wan2.2-TI2V-5B/"
else
  echo "WARN local Wan2.2 missing at ${LOCAL_WAN22}, remote will download via modelscope"
fi

echo "[5/5] remote conda env + link tokenizer"
ssh_cmd "FASTWAM_ROOT=${REMOTE_FASTWAM} CONDA_ENV=fastwam bash ${REMOTE_ROOT}/recipes/setup_env.sh"
ssh_cmd "source ${REMOTE_FASTWAM}/env.sh && FASTWAM_ROOT=${REMOTE_FASTWAM} LOCAL_WAN22=${REMOTE_FASTWAM}/checkpoints/Wan-AI/Wan2.2-TI2V-5B bash ${REMOTE_ROOT}/recipes/link_pretrained.sh"

echo "[OK] gpu-node1 FastWAM ready: ${REMOTE_FASTWAM}"
echo "  ssh ${HOST}"
echo "  source ${REMOTE_FASTWAM}/env.sh"
