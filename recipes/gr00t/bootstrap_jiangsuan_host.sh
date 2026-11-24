#!/bin/bash
# 江算-4 / 江算-1：K8s Pod，无 docker。
#   HOST=jiangsuan-4 bash recipes/gr00t/bootstrap_jiangsuan_host.sh
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
REMOTE_ROOT="${REMOTE_ROOT:-/nvme2n1/gr00t}"
LOCAL_SRC="${LOCAL_SRC:-/home/xk/VLA/Isaac-GR00T}"

ssh_base=(ssh -F "${SSH_CFG}" -o ConnectTimeout=25 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [[ -n "${JUMP}" ]]; then
  ssh_base+=(-o "ProxyJump=${JUMP}")
fi
ssh_cmd() { "${ssh_base[@]}" "${HOST}" "$@"; }

echo "[1/3] tar code/src -> ${HOST}:${REMOTE_ROOT}"
ssh_cmd "mkdir -p ${REMOTE_ROOT}/code ${REMOTE_ROOT}/pretrained ${REMOTE_ROOT}/outputs ${REMOTE_ROOT}/hf ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/src/Isaac-GR00T ${REMOTE_ROOT}/shim"
tar -C "${HERE}" --exclude '.git' -cf - . | "${ssh_base[@]}" "${HOST}" "tar -C '${REMOTE_ROOT}/code' -xf -"
tar -C "${LOCAL_SRC}" --exclude '.git' --exclude '.venv' --exclude '.conda' \
  --exclude 'checkpoints' --exclude 'wheels' --exclude 'runnings' --exclude 'nvidia' \
  --exclude '*.egg-info' -cf - . \
  | "${ssh_base[@]}" "${HOST}" "tar -C '${REMOTE_ROOT}/src/Isaac-GR00T' -xf -"

echo "[2/3] venv on host"
ssh_cmd "bash -lc '
set -euo pipefail
set +u
source /usr/local/PPU_SDK/envsetup.sh
set -u
export ROOT=${REMOTE_ROOT} VENV=${REMOTE_ROOT}/venv SRC=${REMOTE_ROOT}/src/Isaac-GR00T
bash ${REMOTE_ROOT}/code/env/setup_ppu.sh
'"

echo "[3/3] smoke CPU"
ssh_cmd "bash -lc '
set +u
source /usr/local/PPU_SDK/envsetup.sh
source ${REMOTE_ROOT}/venv/bin/activate
export ROOT=${REMOTE_ROOT} SRC=${REMOTE_ROOT}/src/Isaac-GR00T PYTHONPATH=${REMOTE_ROOT}/src/Isaac-GR00T SMOKE_DEV=0
python ${REMOTE_ROOT}/code/env/smoke_test.py
'"
echo "[OK] GR00T host env ready on ${HOST} root=${REMOTE_ROOT} (no docker)"
