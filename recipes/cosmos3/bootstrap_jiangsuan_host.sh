#!/bin/bash
# 江算-4 / 江算-1：K8s Pod，无 docker。目录对齐 16 卡容器 /workspace。
#   HOST=jiangsuan-4 bash recipes/cosmos3/bootstrap_jiangsuan_host.sh
# 不碰正在跑的训练。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SSH_CFG:-}" && -f "${SSH_CFG}" ]]; then
  :
elif [[ -f "${HOME}/.ssh/config" ]]; then
  SSH_CFG="${HOME}/.ssh/config"
elif [[ -f /home/xk/VLA/ssh_config ]]; then
  SSH_CFG=/home/xk/VLA/ssh_config
else
  SSH_CFG="${HOME}/.ssh/config"
fi
HOST="${HOST:-jiangsuan-4}"
JUMP="${JUMP:-gpu-node1}"
REMOTE_ROOT="${REMOTE_ROOT:-/nvme2n1/cosmos3}"
LOCAL_SRC="${LOCAL_SRC:-/home/xk/VLA/cosmos-framework}"

ssh_base=(ssh -F "${SSH_CFG}" -o ConnectTimeout=25 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [[ -n "${JUMP}" ]]; then
  ssh_base+=(-o "ProxyJump=${JUMP}")
fi
ssh_cmd() { "${ssh_base[@]}" "${HOST}" "$@"; }

echo "[1/3] tar code/src -> ${HOST}:${REMOTE_ROOT}"
ssh_cmd "mkdir -p ${REMOTE_ROOT}/code ${REMOTE_ROOT}/pretrained ${REMOTE_ROOT}/outputs ${REMOTE_ROOT}/hf ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/src/cosmos-framework ${REMOTE_ROOT}/shim"
tar -C "${HERE}" --exclude '.git' -cf - . | "${ssh_base[@]}" "${HOST}" "tar -C '${REMOTE_ROOT}/code' -xf -"
tar -C "${LOCAL_SRC}" --exclude '.git' --exclude '.venv' -cf - . \
  | "${ssh_base[@]}" "${HOST}" "tar -C '${REMOTE_ROOT}/src/cosmos-framework' -xf -"

echo "[2/3] venv on host (system-site-packages, keep PPU torch)"
ssh_cmd "bash -lc '
set -euo pipefail
set +u
source /usr/local/PPU_SDK/envsetup.sh
set -u
export ROOT=${REMOTE_ROOT} VENV=${REMOTE_ROOT}/venv CODE=${REMOTE_ROOT}/code SRC=${REMOTE_ROOT}/src/cosmos-framework
bash ${REMOTE_ROOT}/code/env/setup_ppu.sh
'"

echo "[3/3] smoke CPU"
ssh_cmd "bash -lc '
set +u
source /usr/local/PPU_SDK/envsetup.sh
source ${REMOTE_ROOT}/venv/bin/activate
export ROOT=${REMOTE_ROOT} SRC=${REMOTE_ROOT}/src/cosmos-framework PYTHONPATH=${REMOTE_ROOT}/src/cosmos-framework I4_ATTN_BACKENDS=sdpa SMOKE_DEV=0
python ${REMOTE_ROOT}/code/env/smoke_test.py
'"
echo "[OK] Cosmos3 host env ready on ${HOST} root=${REMOTE_ROOT} (no docker)"
