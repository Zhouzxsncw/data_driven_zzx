#!/bin/bash
# 天翼云 cty-18 / cty-19 宿主机：独立 τ0-WM 容器。
# 必须用本机 openpi_train:asllm-1.7.2 + ppu0015 shim，不要套江算 sail 1.6.1 镜像。
# 不碰 openpi_train / wall_oss。树在 /data/extra/tau0wm（3.5T 盘）。
# RECREATE=1 才删容器重建（数据目录保留）。
set -euo pipefail

IMAGE="${IMAGE:-openpi_train:asllm-1.7.2}"
NAME="${NAME:-tau0wm}"
HOST_ROOT="${HOST_ROOT:-/data/extra/tau0wm}"
RECREATE="${RECREATE:-0}"

mkdir -p "${HOST_ROOT}/pretrained" "${HOST_ROOT}/outputs" "${HOST_ROOT}/code" \
         "${HOST_ROOT}/hf" "${HOST_ROOT}/logs" "${HOST_ROOT}/repo" "${HOST_ROOT}/shim"

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

echo "[INFO] creating ${NAME} from ${IMAGE} root=${HOST_ROOT}"
docker run -d \
  --name "${NAME}" \
  --network host \
  --privileged \
  --ipc host \
  --cgroupns host \
  --pids-limit -1 \
  -v "${HOST_ROOT}:/workspace" \
  -v /data/openpi:/data/openpi:ro \
  -v /usr/local/PPU_SDK:/opt/ppu213:ro \
  -v /opt/ppucompat:/opt/ppucompat:ro \
  -v /opt/hggcshim:/opt/hggcshim:ro \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  -e HF_HUB_DISABLE_XET=1 \
  -e HF_HOME=/workspace/hf \
  -e TAU_ROOT=/workspace \
  -w /workspace \
  --restart unless-stopped \
  "${IMAGE}" \
  sleep infinity

docker exec "${NAME}" bash -lc 'cat > /etc/profile.d/zz_ppucompat.sh << "EOF"
export LD_PRELOAD=/opt/hggcshim/libhggcrt1.so
export LD_LIBRARY_PATH=/opt/ppu213/CUDA_SDK/extras/CUPTI/lib64:/usr/local/PPU_SDK/CUDA_SDK/extras/CUPTI/lib64:/opt/ppucompat:/opt/hggcshim:/opt/ppu213/lib:/opt/ppu213/CUDA_SDK/lib64:/opt/ppu213/CUDA_SDK/targets/x86_64-linux/lib
export PPU_SDK=/opt/ppu213
export PPU_PATH=/opt/ppu213
export PPU_HOME=/opt/ppu213
export CUDA_PATH=/opt/ppu213/CUDA_SDK
export CUDA_HOME=/opt/ppu213/CUDA_SDK
export CUDA_TOOLKIT_ROOT=/opt/ppu213/CUDA_SDK
EOF
HN=$(hostname)
grep -q "$HN" /etc/hosts || echo "127.0.0.1 ${HN}" >> /etc/hosts
echo PROFILE_OK'

# 镜像自带的 torch 是 ppu001，在 ppu0015 上 import 会崩。装宿主机上的 ppu2.1.0 wheel。
docker exec "${NAME}" bash -lc 'source /etc/profile.d/zz_ppucompat.sh
  python3 -m pip install --no-deps /data/openpi/torch-*.whl /data/openpi/torchvision-*.whl
  python3 -c "import torch; print(\"cty_torch\", torch.__version__, torch.cuda.is_available())"'

echo "[OK] ${NAME} up"
docker ps --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
echo "enter: docker exec -it ${NAME} bash"
echo "setup: docker exec ${NAME} bash -lc \"source /etc/profile.d/zz_ppucompat.sh; bash /workspace/code/docker/setup_in_container.sh\""
