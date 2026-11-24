#!/bin/bash
# 华为 Atlas：独立 cosmos3 容器。复用 pi05/wall-oss 同款 vllm-ascend 镜像，不碰它们。
# A3 910C×16: IMAGE=...:v0.22.1rc1-a3  RUNTIME=runc
# A2 910B×8 : IMAGE=...:v0.22.1rc1     RUNTIME=ascend
# RECREATE=1 删容器重建（/home/cosmos3 保留）。
set -euo pipefail

NAME="${NAME:-cosmos3}"
HOST_ROOT="${HOST_ROOT:-/home/cosmos3}"
RECREATE="${RECREATE:-0}"

npu_count() {
  ls -1 /dev/davinci[0-9]* 2>/dev/null | grep -E '/dev/davinci[0-9]+$' | wc -l
}

NPU_N="${NPU_N:-$(npu_count)}"
if [ -e /dev/davinci8 ] || [ "${NPU_N}" -ge 16 ]; then
  IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.22.1rc1-a3}"
  RUNTIME="${RUNTIME:-runc}"
elif docker info 2>/dev/null | grep -qi 'ascend'; then
  IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.22.1rc1}"
  RUNTIME="${RUNTIME:-ascend}"
else
  IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.22.1rc1}"
  RUNTIME="${RUNTIME:-runc}"
fi

mkdir -p "${HOST_ROOT}/pretrained" "${HOST_ROOT}/outputs" "${HOST_ROOT}/src" \
         "${HOST_ROOT}/hf" "${HOST_ROOT}/logs" "${HOST_ROOT}/shim" "${HOST_ROOT}/code"

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

DEVS=()
for i in $(seq 0 $((NPU_N - 1))); do
  [ -e "/dev/davinci${i}" ] && DEVS+=(--device "/dev/davinci${i}")
done
for d in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [ -e "$d" ] && DEVS+=(--device "$d")
done

BINDS=(-v "${HOST_ROOT}:${HOST_ROOT}"
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
       -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons
       -v /usr/local/Ascend/dcmi:/usr/local/Ascend/dcmi)
[ -e /usr/local/bin/npu-smi ] && BINDS+=(-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi)
[ -e /etc/vnpu.cfg ] && BINDS+=(-v /etc/vnpu.cfg:/etc/vnpu.cfg:ro)
if [ -d /home/software ]; then
  mkdir -p "/home/software/${NAME}/outputs"
  BINDS+=(-v /home/software:/home/software)
fi

RT_ARGS=()
[ -n "${RUNTIME}" ] && [ "${RUNTIME}" != "runc" ] && RT_ARGS+=(--runtime "${RUNTIME}")

echo "[INFO] creating ${NAME} image=${IMAGE} runtime=${RUNTIME} npus=${NPU_N}"
docker run -d \
  --name "${NAME}" \
  --privileged \
  --net=host \
  --ipc=host \
  --shm-size=64g \
  "${RT_ARGS[@]}" \
  "${BINDS[@]}" \
  "${DEVS[@]}" \
  --restart unless-stopped \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -c 'for f in /usr/local/Ascend/ascend-toolkit/set_env.sh \
               /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh \
               /usr/local/Ascend/nnal/atb/set_env.sh; do
        [ -f "$f" ] && source "$f"
      done
      exec tail -f /dev/null'

echo "[OK] ${NAME} up"
docker ps --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
echo "enter: docker exec -it ${NAME} bash"
echo "root:  ${HOST_ROOT}"
