#!/bin/bash
# 本机跑：把 WALL-OSS 代码/源码/预训练装到 jiangsuan-4。
#   bash recipes/walloss/bootstrap_jiangsuan4.sh
# 4 卡机是 K8s Pod，没有 docker；目录对齐 16 卡容器：
#   宿主机 /nvme2n1/wall-oss  ≡  16 卡 /nvme1n1/wall-oss → /workspace
# 直连 SSH 常被掐，默认经 gpu-node1 跳板。不碰正在跑的 LingBot-VLA。
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
REMOTE_ROOT="${REMOTE_ROOT:-/nvme2n1/wall-oss}"
SRC16="${SRC16:-jiangsuan-16:/nvme1n1/wall-oss/pretrained/wall-oss-0.5}"

ssh_base=(ssh -F "${SSH_CFG}" -o ConnectTimeout=25 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [[ -n "${JUMP}" ]]; then
  ssh_base+=(-o "ProxyJump=${JUMP}")
fi

ssh_cmd() {
  "${ssh_base[@]}" "${HOST}" "$@"
}

VENDOR="${VENDOR:-${HERE}/.vendor}"
if [[ ! -f "${VENDOR}/wall-x/setup.py" ]] || [[ ! -f "${VENDOR}/lerobot/pyproject.toml" ]]; then
  echo "[0/5] missing ${VENDOR}; run recipes/walloss/bootstrap_jiangsuan16.sh once to fetch wall-x/lerobot" >&2
  exit 1
fi

echo "[1/5] mkdir + tar code/src -> ${HOST}:${REMOTE_ROOT}"
ssh_cmd "mkdir -p ${REMOTE_ROOT}/code ${REMOTE_ROOT}/pretrained ${REMOTE_ROOT}/outputs ${REMOTE_ROOT}/hf ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/src/wall-x ${REMOTE_ROOT}/src/lerobot ${REMOTE_ROOT}/datasets"

# 对端没有 rsync
tar -C "${HERE}" --exclude '.git' --exclude '.vendor' -cf - . \
  | "${ssh_base[@]}" "${HOST}" "tar -C '${REMOTE_ROOT}/code' -xf -"
tar -C /home/xk/data_Driven/scripts -cf - hw_train_monitor.py start_hw_monitor.sh \
  | "${ssh_base[@]}" "${HOST}" "tar -C '${REMOTE_ROOT}/code/train' -xf -"
tar -C "${VENDOR}/wall-x" --exclude '.git' -cf - . \
  | "${ssh_base[@]}" "${HOST}" "tar -C '${REMOTE_ROOT}/src/wall-x' -xf -"
tar -C "${VENDOR}/lerobot" --exclude '.git' -cf - . \
  | "${ssh_base[@]}" "${HOST}" "tar -C '${REMOTE_ROOT}/src/lerobot' -xf -"

echo "[2/5] pretrained wall-oss-0.5 (copy from jiangsuan-16 if missing)"
if ssh_cmd "test -f ${REMOTE_ROOT}/pretrained/wall-oss-0.5/model.safetensors"; then
  echo "[INFO] pretrained already present"
else
  echo "[INFO] tar stream ${SRC16} -> ${HOST}:${REMOTE_ROOT}/pretrained/"
  "${ssh_base[@]}" jiangsuan-16 "tar -C /nvme1n1/wall-oss/pretrained -cf - wall-oss-0.5" \
    | "${ssh_base[@]}" "${HOST}" "tar -C '${REMOTE_ROOT}/pretrained' -xf -"
fi

echo "[3/5] venv on host (system-site-packages, keep PPU torch)"
ssh_cmd "bash -lc '
set -euo pipefail
set +u
source /usr/local/PPU_SDK/envsetup.sh
set -u
export ROOT=${REMOTE_ROOT}
export VENV=${REMOTE_ROOT}/venv
export CODE=${REMOTE_ROOT}/code
export WALLX=${REMOTE_ROOT}/src/wall-x
export LEROBOT=${REMOTE_ROOT}/src/lerobot
export CONSTRAINTS=${REMOTE_ROOT}/code/env/constraints_walloss_host.txt
export WALLX_SKIP_CUDA_EXT=1
bash ${REMOTE_ROOT}/code/env/setup_walloss.sh
'"

echo "[4/5] verify (CPU load only; do not steal VLA GPUs)"
ssh_cmd "bash -lc '
set -euo pipefail
set +u
source /usr/local/PPU_SDK/envsetup.sh
source ${REMOTE_ROOT}/venv/bin/activate
set -u
export ROOT=${REMOTE_ROOT}
export CKPT=${REMOTE_ROOT}/pretrained/wall-oss-0.5
export CUDA_VISIBLE_DEVICES=
bash ${REMOTE_ROOT}/code/train/verify_load.sh
'"

echo "[OK] WALL-OSS host env ready on ${HOST}"
echo "  ssh ${HOST}  # may need ProxyJump=${JUMP}"
echo "  source /usr/local/PPU_SDK/envsetup.sh && source ${REMOTE_ROOT}/venv/bin/activate"
echo "  ckpt: ${REMOTE_ROOT}/pretrained/wall-oss-0.5"
