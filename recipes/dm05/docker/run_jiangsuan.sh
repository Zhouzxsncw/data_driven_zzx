#!/bin/bash
# 江算宿主机：独立 DM0.5 容器。复用 PPU asllm 镜像。
# 不挂 /dev/alixpu*，不用 --privileged / --gpus。
# 不碰 pytorch-ppu / openpi_train / cosmos3 / wall_oss。
# RECREATE=1 才删容器（/opt/opendm 保留）。
set -euo pipefail

IMAGE="${IMAGE:-cr-ee.registry.cn-hami-jsry-d01.res.cloud.jsry.com/ppu/asllm:1.7.2-pytorch2.7.0-ubuntu24.04-sail1.6.1-cuda12.8-vllm0.10.0-sglang0.5.2-py312}"
NAME="${NAME:-dm05}"
HOST_ROOT="${HOST_ROOT:-/opt/opendm}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace}"
RECREATE="${RECREATE:-0}"

mkdir -p "${HOST_ROOT}/pretrained" "${HOST_ROOT}/src" "${HOST_ROOT}/code" "${HOST_ROOT}/logs" "${HOST_ROOT}/hf"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[ERR] image missing: ${IMAGE}" >&2
  echo "  这台仓库证书对不上，不要 docker pull，也不要改 daemon.json（会重启 docker，停掉正在跑的容器）。" >&2
  echo "  从已有镜像的 16b 低优先级流过来：ionice -c 3 nice -n 19 docker save | docker load" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  if [ "${RECREATE}" != "1" ]; then
    echo "[INFO] container ${NAME} already exists"
    docker start "${NAME}" >/dev/null
  else
    echo "[INFO] RECREATE=1 removing ${NAME} (volume ${HOST_ROOT} kept)"
    docker rm -f "${NAME}" >/dev/null
  fi
fi

if ! docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "[INFO] creating ${NAME} from ${IMAGE} root=${HOST_ROOT} (no PPU devices)"
  docker run -d \
    --name "${NAME}" \
    --runtime=runc \
    --net=host \
    --ipc=host \
    --shm-size=8g \
    -v "${HOST_ROOT}:${CONTAINER_ROOT}" \
    -e CUDA_VISIBLE_DEVICES= \
    -e PPU_VISIBLE_DEVICES= \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HOME="${CONTAINER_ROOT}/hf" \
    --restart unless-stopped \
    "${IMAGE}" \
    sleep infinity
fi

echo "[OK] ${NAME} up"
docker ps --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
# 有的 PPU 宿主会把父设备 /dev/alixpu 自动放进容器，但不会放 alixpu_ppu*。
# 只删容器自己的 /dev 节点，不碰宿主。父设备留着时 torch 仍可能去 open 它。
docker exec "${NAME}" bash -lc 'rm -f /dev/alixpu /dev/alixpu_ctl /dev/alixpu_ppu* /dev/alixpu_sep; rm -rf /dev/alixpu-caps'
if docker exec "${NAME}" bash -lc 'compgen -G "/dev/alixpu*" >/dev/null'; then
  echo "[ERR] ${NAME} still has /dev/alixpu*. Do not smoke." >&2
  exit 1
fi
echo "[OK] no /dev/alixpu inside ${NAME}"
echo "enter: docker exec -it ${NAME} bash"
echo "setup: docker exec ${NAME} bash ${CONTAINER_ROOT}/code/env/setup_ppu.sh"
