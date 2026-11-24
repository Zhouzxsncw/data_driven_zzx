#!/bin/bash
# jiangsuan-16 宿主机：独立 WALL-OSS 容器。
# 复用江算 PPU 镜像，不碰 vlm_sft / lingbotva_train / lingbotvla_train / openpi_train。
#
# RECREATE=1 删容器重建（/nvme1n1/wall-oss 数据保留）。
set -euo pipefail

IMAGE="${IMAGE:-cr-ee.registry.cn-hami-jsry-d01.res.cloud.jsry.com/ppu/asllm:1.7.2-pytorch2.7.0-ubuntu24.04-sail1.6.1-cuda12.8-vllm0.10.0-sglang0.5.2-py312}"
NAME="${NAME:-wall_oss}"
HOST_ROOT="${HOST_ROOT:-/nvme1n1/wall-oss}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace}"
RECREATE="${RECREATE:-0}"

mkdir -p "${HOST_ROOT}/pretrained" "${HOST_ROOT}/outputs" "${HOST_ROOT}/code" \
         "${HOST_ROOT}/hf" "${HOST_ROOT}/logs"

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  if [ "${RECREATE}" != "1" ]; then
    echo "[INFO] container ${NAME} already exists"
    docker start "${NAME}" >/dev/null
    docker ps --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
    exit 0
  fi
  echo "[INFO] RECREATE=1 removing ${NAME} (volume ${HOST_ROOT} kept)"
  docker rm -f "${NAME}" >/dev/null
fi

echo "[INFO] creating ${NAME} from ${IMAGE}"
docker run -d \
  --name "${NAME}" \
  --privileged \
  --net=host \
  --ipc=host \
  --security-opt label=disable \
  --shm-size=32g \
  -v "${HOST_ROOT}:${CONTAINER_ROOT}" \
  -v /nvme1n1/openpi/datasets:/data/datasets:ro \
  -v /opt/pg1:/opt/pg1 \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  -e HF_HOME="${CONTAINER_ROOT}/hf" \
  -e MODELSCOPE_CACHE="${CONTAINER_ROOT}/hf/modelscope" \
  -e TRANSFORMERS_OFFLINE=0 \
  --restart unless-stopped \
  "${IMAGE}" \
  sleep infinity

echo "[OK] ${NAME} up"
docker ps --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
echo "enter:  docker exec -it ${NAME} bash"
echo "setup:  docker exec ${NAME} bash ${CONTAINER_ROOT}/code/env/setup_walloss.sh"
echo "ckpt:   docker exec ${NAME} bash ${CONTAINER_ROOT}/code/download/download_walloss.sh"
