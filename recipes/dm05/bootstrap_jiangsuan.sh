#!/bin/bash
# 本机跑：同步 OpenDM 代码和 DM05 预训练，起独立容器，CPU smoke。
#   HOST=jiangsuan-16 bash recipes/dm05/bootstrap_jiangsuan.sh
# 不杀已有进程。dm05 容器不挂 /dev/alixpu*。不传 user_checkpoints。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
if [[ -n "${SSH_CFG:-}" && -f "${SSH_CFG}" ]]; then
  :
elif [[ -f "${HOME}/.ssh/config" ]]; then
  SSH_CFG="${HOME}/.ssh/config"
else
  SSH_CFG="${HOME}/.ssh/config"
fi
HOST="${HOST:-jiangsuan-16}"
REMOTE_ROOT="${REMOTE_ROOT:-/opt/opendm}"
NAME="${NAME:-dm05}"
LOCAL_SRC="${LOCAL_SRC:-/home/xk/VLA/opendm}"
LOCAL_CKPT="${LOCAL_CKPT:-${LOCAL_SRC}/checkpoints/DM05}"
EXPECT_SAFETENSORS="${EXPECT_SAFETENSORS:-11658431136}"

ssh_cmd() {
  ssh -F "${SSH_CFG}" -o BatchMode=yes -o ConnectTimeout=20 "${HOST}" "$@"
}
RSYNC_SSH="ssh -F ${SSH_CFG} -o BatchMode=yes"

echo "[1/5] rsync code -> ${HOST}:${REMOTE_ROOT}"
ssh_cmd "mkdir -p ${REMOTE_ROOT}/code ${REMOTE_ROOT}/src/opendm ${REMOTE_ROOT}/pretrained/DM05 ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/hf"
rsync -az -e "${RSYNC_SSH}" \
  --exclude '.git' \
  "${HERE}/" "${HOST}:${REMOTE_ROOT}/code/"
rsync -az -e "${RSYNC_SSH}" \
  --exclude '.git' --exclude 'checkpoints' --exclude 'user_checkpoints' \
  --exclude 'wandb' --exclude 'data' --exclude '__pycache__' --exclude '*.egg-info' \
  "${LOCAL_SRC}/" "${HOST}:${REMOTE_ROOT}/src/opendm/"

echo "[2/5] rsync DM05 pretrained (skip when size matches)"
remote_size="$(ssh_cmd "stat -c %s ${REMOTE_ROOT}/pretrained/DM05/model.safetensors 2>/dev/null || echo 0")"
if [[ "${remote_size}" == "${EXPECT_SAFETENSORS}" ]]; then
  echo "[INFO] model.safetensors already ${EXPECT_SAFETENSORS}"
else
  rsync -a --partial --info=progress2 -e "${RSYNC_SSH}" \
    "${LOCAL_CKPT}/" "${HOST}:${REMOTE_ROOT}/pretrained/DM05/"
fi

echo "[3/5] docker ${NAME} (no PPU devices)"
ssh_cmd "HOST_ROOT=${REMOTE_ROOT} NAME=${NAME} bash ${REMOTE_ROOT}/code/docker/run_jiangsuan.sh"

echo "[4/5] env (keep image torch, no flash-attn)"
ssh_cmd "docker exec -e CUDA_VISIBLE_DEVICES= ${NAME} bash /workspace/code/env/setup_ppu.sh"

echo "[5/5] CPU smoke"
ssh_cmd "docker exec -e CUDA_VISIBLE_DEVICES= -e ROOT=/workspace -e DM05_CKPT=/workspace/pretrained/DM05 ${NAME} bash -lc 'source /workspace/venv/bin/activate; python /workspace/code/env/smoke_test.py'"

echo "[OK] DM0.5 docker ready on ${HOST} name=${NAME} root=${REMOTE_ROOT}"
