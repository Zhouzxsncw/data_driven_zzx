#!/bin/bash
# H100 宿主机：独立 tau0wm 容器，挂已有 /data1/tau0wm（不删 envs/、outputs/、pretrained/）。
# 本机历史上没有 docker.service。缺服务时只装 docker.io，不 apt upgrade、不重启训练。
# RECREATE=1 只删容器，不删 /data1/tau0wm。
set -euo pipefail

NAME="${NAME:-tau0wm}"
HOST_ROOT="${HOST_ROOT:-/data1/tau0wm}"
IMAGE="${IMAGE:-sealos.hub:5000/sglang:v0.5.6-cu129-amd64-awcloud}"
RECREATE="${RECREATE:-0}"

if ! sudo -n systemctl is-active docker >/dev/null 2>&1; then
  if ! systemctl cat docker.service >/dev/null 2>&1; then
    echo "[INFO] 安装 docker.io（不 upgrade）"
    if sudo -n apt-get install -s docker.io 2>/dev/null | grep -E '^Remv ' | grep -qi 'nvidia'; then
      echo "[FATAL] 安装 docker.io 会卸掉 nvidia 包，已停下" >&2
      exit 1
    fi
    sudo -n apt-get update -qq
    sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
  fi
  sudo -n systemctl enable --now docker
fi
sudo -n docker info >/dev/null

mkdir -p "${HOST_ROOT}/pretrained" "${HOST_ROOT}/outputs" "${HOST_ROOT}/code" \
         "${HOST_ROOT}/hf" "${HOST_ROOT}/logs" "${HOST_ROOT}/repo" "${HOST_ROOT}/shim"

if ! sudo -n docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[INFO] pull ${IMAGE}"
  sudo -n docker pull "${IMAGE}"
fi

if sudo -n docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  if [ "${RECREATE}" != "1" ]; then
    echo "[INFO] container ${NAME} already exists"
    sudo -n docker start "${NAME}" >/dev/null
    sudo -n docker ps --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
    exit 0
  fi
  echo "[INFO] RECREATE=1 removing ${NAME} (volume ${HOST_ROOT} kept)"
  sudo -n docker rm -f "${NAME}" >/dev/null
fi

echo "[INFO] creating ${NAME} from ${IMAGE} root=${HOST_ROOT}"
sudo -n docker run -d \
  --name "${NAME}" \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size=32g \
  -v "${HOST_ROOT}:/workspace" \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  -e HF_HUB_DISABLE_XET=1 \
  -e HF_HOME=/workspace/hf \
  -e TAU_ROOT=/workspace \
  -w /workspace \
  --restart unless-stopped \
  "${IMAGE}" \
  sleep infinity

echo "[OK] ${NAME} up"
sudo -n docker ps --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
