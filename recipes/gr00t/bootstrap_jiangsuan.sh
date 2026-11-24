#!/bin/bash
# 本机跑：同步 GR00T 源码（不含 ckpt/.conda）、起独立容器、装环境、CPU smoke。
#   HOST=jiangsuan-16 bash recipes/gr00t/bootstrap_jiangsuan.sh
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
HOST="${HOST:-jiangsuan-16}"
REMOTE_ROOT="${REMOTE_ROOT:-/nvme2n1/gr00t}"
NAME="${NAME:-gr00t}"
LOCAL_SRC="${LOCAL_SRC:-/home/xk/VLA/Isaac-GR00T}"

ssh_cmd() {
  ssh -F "${SSH_CFG}" -o ConnectTimeout=20 -o BatchMode=yes "${HOST}" "$@"
}

echo "[1/4] rsync code + src -> ${HOST}:${REMOTE_ROOT}"
ssh_cmd "mkdir -p ${REMOTE_ROOT}/code ${REMOTE_ROOT}/pretrained ${REMOTE_ROOT}/outputs ${REMOTE_ROOT}/hf ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/src ${REMOTE_ROOT}/shim"
rsync -az -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  --exclude '.git' "${HERE}/" "${HOST}:${REMOTE_ROOT}/code/"
rsync -az -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  --exclude '.git' --exclude '.venv' --exclude '.conda' \
  --exclude 'checkpoints' --exclude 'wheels' --exclude 'runnings' \
  --exclude 'nvidia' --exclude '__pycache__' --exclude '*.egg-info' \
  "${LOCAL_SRC}/" "${HOST}:${REMOTE_ROOT}/src/Isaac-GR00T/"

echo "[2/4] docker ${NAME}"
ssh_cmd "HOST_ROOT=${REMOTE_ROOT} NAME=${NAME} bash ${REMOTE_ROOT}/code/docker/run_jiangsuan.sh"

echo "[3/4] env (keep image torch, skip flash-attn/tensorrt)"
ssh_cmd "docker exec ${NAME} bash /workspace/code/env/setup_ppu.sh"

echo "[4/4] smoke"
ssh_cmd "docker exec -e SMOKE_DEV=0 ${NAME} bash -lc '
  source /workspace/venv/bin/activate
  export PYTHONPATH=/workspace/src/Isaac-GR00T:/workspace/shim ROOT=/workspace SRC=/workspace/src/Isaac-GR00T
  python /workspace/code/env/smoke_test.py
'"

echo "[OK] GR00T docker ready on ${HOST} name=${NAME} root=${REMOTE_ROOT}"
echo "  docker exec -it ${NAME} bash"
echo "  pretrained/backbone not copied; copy later if training."
