#!/bin/bash
# 在本机跑：同步本仓 recipes/walloss、起容器、装环境、下 WALL-OSS-0.5。
#   bash recipes/walloss/bootstrap_jiangsuan16.sh
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
REMOTE_ROOT="${REMOTE_ROOT:-/nvme1n1/wall-oss}"
NAME="${NAME:-wall_oss}"

ssh_cmd() {
  ssh -F "${SSH_CFG}" -o ConnectTimeout=20 -o BatchMode=yes "${HOST}" "$@"
}

VENDOR="${VENDOR:-${HERE}/.vendor}"
mkdir -p "${VENDOR}"
if [ ! -f "${VENDOR}/wall-x/setup.py" ]; then
  echo "[0/5] fetch qiaow371/wall-x tongbot (not upstream main)"
  if [ -f /home/xk/VLA/wall-x/setup.py ]; then
    mkdir -p "${VENDOR}/wall-x"
    rsync -a --exclude '.git' /home/xk/VLA/wall-x/ "${VENDOR}/wall-x/"
  elif [ -f /data/wall-oss/src/wall-x/setup.py ]; then
    mkdir -p "${VENDOR}/wall-x"
    rsync -a --exclude '.git' /data/wall-oss/src/wall-x/ "${VENDOR}/wall-x/"
  elif git clone --depth 1 -b tongbot https://github.com/qiaow371/wall-x.git "${VENDOR}/wall-x"; then
    :
  else
    mkdir -p "${VENDOR}/wall-x"
    tmpd="$(mktemp -d)"
    curl -fL --retry 3 \
      "https://ghfast.top/https://github.com/qiaow371/wall-x/archive/refs/heads/tongbot.tar.gz" \
      -o "${tmpd}/wall-x.tgz"
    tar -xzf "${tmpd}/wall-x.tgz" -C "${VENDOR}/wall-x" --strip-components=1
    rm -rf "${tmpd}"
  fi
fi
if [ ! -f "${VENDOR}/lerobot/pyproject.toml" ]; then
  echo "[0/5] fetch huggingface/lerobot"
  if [ -f /data/wall-oss/src/lerobot/pyproject.toml ]; then
    mkdir -p "${VENDOR}/lerobot"
    rsync -a --exclude '.git' /data/wall-oss/src/lerobot/ "${VENDOR}/lerobot/"
  else
    mkdir -p "${VENDOR}/lerobot"
    tmpd="$(mktemp -d)"
    curl -fL --retry 3 \
      "https://ghfast.top/https://github.com/huggingface/lerobot/archive/refs/heads/main.tar.gz" \
      -o "${tmpd}/lerobot.tgz"
    tar -xzf "${tmpd}/lerobot.tgz" -C "${VENDOR}/lerobot" --strip-components=1
    rm -rf "${tmpd}"
  fi
fi

echo "[1/5] rsync code + src -> ${HOST}:${REMOTE_ROOT}"
ssh_cmd "mkdir -p ${REMOTE_ROOT}/code ${REMOTE_ROOT}/pretrained ${REMOTE_ROOT}/outputs ${REMOTE_ROOT}/hf ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/src"
rsync -az \
  -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  --exclude '.git' --exclude '.vendor' \
  "${HERE}/" "${HOST}:${REMOTE_ROOT}/code/"
rsync -az \
  -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  /home/xk/data_Driven/scripts/hw_train_monitor.py \
  /home/xk/data_Driven/scripts/start_hw_monitor.sh \
  "${HOST}:${REMOTE_ROOT}/code/train/"
rsync -az \
  -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  --exclude '.git' \
  "${VENDOR}/wall-x/" "${HOST}:${REMOTE_ROOT}/src/wall-x/"
rsync -az \
  -e "ssh -F ${SSH_CFG} -o BatchMode=yes" \
  --exclude '.git' \
  "${VENDOR}/lerobot/" "${HOST}:${REMOTE_ROOT}/src/lerobot/"

echo "[2/5] docker"
ssh_cmd "bash ${REMOTE_ROOT}/code/docker/run_walloss_16gpu.sh"

echo "[3/5] env (wall-x + lerobot, skip CUDA ext)"
ssh_cmd "docker exec ${NAME} bash /workspace/code/env/setup_walloss.sh"

echo "[4/5] download wall-oss-0.5"
ssh_cmd "docker exec ${NAME} bash /workspace/code/download/download_walloss.sh"

echo "[5/5] verify"
ssh_cmd "docker exec ${NAME} bash /workspace/code/train/verify_load.sh"

echo "[OK] WALL-OSS docker ready on ${HOST}"
echo "  docker exec -it ${NAME} bash"
echo "  ckpt: ${REMOTE_ROOT}/pretrained/wall-oss-0.5"
