#!/bin/bash
# 本机跑：同步 Cosmos3 代码、起独立容器、装环境、CPU smoke。
#   HOST=jiangsuan-16 bash recipes/cosmos3/bootstrap_jiangsuan.sh
#   HOST=jiangsuan-16b bash recipes/cosmos3/bootstrap_jiangsuan.sh
# 不杀训练、不碰 openpi_train / wall_oss。不传 33G Nano 预训练。
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
REMOTE_ROOT="${REMOTE_ROOT:-/nvme2n1/cosmos3}"
NAME="${NAME:-cosmos3}"
LOCAL_SRC="${LOCAL_SRC:-/home/xk/VLA/cosmos-framework}"

ssh_cmd() {
  ssh -F "${SSH_CFG}" -o ConnectTimeout=20 -o BatchMode=yes "${HOST}" "$@"
}

echo "[1/4] rsync code + src -> ${HOST}:${REMOTE_ROOT}"
ssh_cmd "mkdir -p ${REMOTE_ROOT}/code ${REMOTE_ROOT}/pretrained ${REMOTE_ROOT}/outputs ${REMOTE_ROOT}/hf ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/src ${REMOTE_ROOT}/shim"
rsync -az -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  --exclude '.git' --exclude 'setup_env.log' \
  "${HERE}/" "${HOST}:${REMOTE_ROOT}/code/"
rsync -az -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  "${LOCAL_SRC}/" "${HOST}:${REMOTE_ROOT}/src/cosmos-framework/"

echo "[2/4] docker ${NAME}"
ssh_cmd "HOST_ROOT=${REMOTE_ROOT} NAME=${NAME} bash ${REMOTE_ROOT}/code/docker/run_jiangsuan.sh"

echo "[3/4] env (skip flash-attn, register sdpa)"
ssh_cmd "docker exec ${NAME} bash /workspace/code/env/setup_ppu.sh"

echo "[4/4] smoke"
ssh_cmd "docker exec -e SMOKE_DEV=0 ${NAME} bash -lc '
  source /workspace/venv/bin/activate
  export PYTHONPATH=/workspace/src/cosmos-framework:/workspace/shim
  export I4_ATTN_BACKENDS=sdpa ROOT=/workspace SRC=/workspace/src/cosmos-framework
  python /workspace/code/env/smoke_test.py
'"

echo "[OK] Cosmos3 docker ready on ${HOST} name=${NAME} root=${REMOTE_ROOT}"
echo "  docker exec -it ${NAME} bash"
echo "  pretrained not copied (Cosmos3-Nano ~33G); copy later if training."
