#!/usr/bin/env bash
# 把 Motus 代码 + 预训练权重推到 gpu-node1，并在远端装环境。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_CFG="${SSH_CFG:-/home/xk/VLA/ssh_config}"
HOST="${HOST:-gpu-node1}"
LOCAL_MOTUS="${LOCAL_MOTUS:-/home/xk/VLA/Motus}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/motus}"
REMOTE_MOTUS="${REMOTE_MOTUS:-${REMOTE_ROOT}/Motus}"
REMOTE_WAN22="${REMOTE_WAN22:-/data/fastwam/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B}"

ssh_cmd() {
  ssh -F "${SSH_CFG}" -o ConnectTimeout=20 -o BatchMode=yes "${HOST}" "$@"
}

echo "[1/5] mkdir ${HOST}:${REMOTE_ROOT}"
ssh_cmd "mkdir -p ${REMOTE_MOTUS}/pretrained_models ${REMOTE_ROOT}/runs ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/recipes"

echo "[2/5] rsync Motus code"
rsync -az --info=progress2 \
  -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' \
  --exclude '.hf' --exclude '.modelscope' --exclude 'outputs' \
  --exclude 'pretrained_models' --exclude 'checkpoints' \
  "${LOCAL_MOTUS}/" "${HOST}:${REMOTE_MOTUS}/"

echo "[3/5] rsync recipes/motus"
rsync -az --info=progress2 \
  -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  "${HERE}/" "${HOST}:${REMOTE_ROOT}/recipes/"

echo "[4/5] rsync Motus / Qwen weights (skip Wan2.2, remote symlink FastWAM)"
if [[ -d "${LOCAL_MOTUS}/pretrained_models" ]]; then
  rsync -a --info=progress2 \
    -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
    --exclude 'Wan2.2-TI2V-5B' \
    "${LOCAL_MOTUS}/pretrained_models/" "${HOST}:${REMOTE_MOTUS}/pretrained_models/"
else
  echo "WARN local pretrained_models missing, remote will download"
fi

echo "[5/5] remote conda env + link pretrained"
ssh_cmd "MOTUS_ROOT=${REMOTE_MOTUS} CONDA_ENV=motus bash ${REMOTE_ROOT}/recipes/setup_env.sh"
ssh_cmd "MOTUS_ROOT=${REMOTE_MOTUS} LOCAL_WAN22=${REMOTE_WAN22} bash ${REMOTE_ROOT}/recipes/link_pretrained.sh"

echo "[OK] ${HOST} Motus ready: ${REMOTE_MOTUS}"
echo "  ssh -F ${SSH_CFG} ${HOST}"
echo "  source ${REMOTE_MOTUS}/env.sh"
