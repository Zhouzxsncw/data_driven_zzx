#!/bin/bash
# 江算 16b/16c 宿主机：独立 τ0-WM 容器。
# 复用 PPU asllm 镜像，不碰 openpi_train / wall_oss / cosmos3 / gr00t / lingbot*。
# 树在 /nvme2n1/tau0wm，容器内 /workspace。
# RECREATE=1 才删容器重建（数据目录保留）。
set -euo pipefail

IMAGE="${IMAGE:-cr-ee.registry.cn-hami-jsry-d01.res.cloud.jsry.com/ppu/asllm:1.7.2-pytorch2.7.0-ubuntu24.04-sail1.6.1-cuda12.8-vllm0.10.0-sglang0.5.2-py312}"
NAME="${NAME:-tau0wm}"
HOST_ROOT="${HOST_ROOT:-/nvme2n1/tau0wm}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace}"
RECREATE="${RECREATE:-0}"

mkdir -p "${HOST_ROOT}/pretrained" "${HOST_ROOT}/outputs" "${HOST_ROOT}/code" \
         "${HOST_ROOT}/hf" "${HOST_ROOT}/logs" "${HOST_ROOT}/repo" "${HOST_ROOT}/shim" \
         "${HOST_ROOT}/datasets"

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

DS_BIND=()
if [ -d /nvme1n1/openpi/datasets ]; then
  DS_BIND+=(-v /nvme1n1/openpi/datasets:/data/datasets:ro)
fi

echo "[INFO] creating ${NAME} from ${IMAGE} root=${HOST_ROOT}"
docker run -d \
  --name "${NAME}" \
  --privileged \
  --net=host \
  --ipc=host \
  --security-opt label=disable \
  --shm-size=32g \
  -v "${HOST_ROOT}:${CONTAINER_ROOT}" \
  "${DS_BIND[@]}" \
  -v /opt/pg1:/opt/pg1 \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  -e HF_HUB_DISABLE_XET=1 \
  -e HF_HOME="${CONTAINER_ROOT}/hf" \
  -e TAU_ROOT="${CONTAINER_ROOT}" \
  --restart unless-stopped \
  "${IMAGE}" \
  sleep infinity

echo "[OK] ${NAME} up"
docker ps --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
echo "enter: docker exec -it ${NAME} bash"
echo "setup: docker exec ${NAME} bash ${CONTAINER_ROOT}/code/docker/setup_in_container.sh"
