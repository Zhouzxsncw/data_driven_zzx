#!/usr/bin/env bash
# 把 Motus 代码 + 预训练权重推到 h100-103，并在远端装环境。
# 需要先连 atrust VPN。
#
# 用法：
#   bash recipes/motus/deploy_h100.sh            # 全流程
#   SKIP_ENV=1 bash recipes/motus/deploy_h100.sh # 跳过远端 conda 安装
#   SKIP_WAN22=1 bash recipes/motus/deploy_h100.sh  # 跳过 Wan2.2 上传（32G）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_CFG="${SSH_CFG:-/home/xk/VLA/ssh_config}"
HOST="${HOST:-h100-103}"
LOCAL_MOTUS="${LOCAL_MOTUS:-/home/xk/VLA/Motus}"
LOCAL_WAN22="${LOCAL_WAN22:-/home/xk/VLA/dreamzero_staging/pretrained/Wan2.2-TI2V-5B}"
REMOTE_ROOT="${REMOTE_ROOT:-/data1/motus}"
REMOTE_MOTUS="${REMOTE_MOTUS:-${REMOTE_ROOT}/Motus}"
SKIP_ENV="${SKIP_ENV:-0}"
SKIP_WAN22="${SKIP_WAN22:-0}"

# h100-103 走 gateway 密码跳板，不能用 BatchMode=yes
ssh_cmd() {
  ssh -F "${SSH_CFG}" -o ConnectTimeout=30 "${HOST}" "$@"
}

rsync_ssh() {
  rsync -az --info=progress2 \
    -e "ssh -F ${SSH_CFG}" \
    "$@"
}

echo "=========================================="
echo "  Motus -> ${HOST} 部署"
echo "  本地: ${LOCAL_MOTUS}"
echo "  远端: ${REMOTE_MOTUS}"
echo "=========================================="

# 连通性检查
echo "[0/6] SSH 连通性检查 ${HOST}..."
if ! ssh_cmd "echo ok" 2>/dev/null; then
  echo "ERROR: 无法连接 ${HOST}，请确认 atrust VPN 已启动" >&2
  exit 1
fi
echo "  SSH OK"

echo "[1/6] mkdir ${HOST}:${REMOTE_ROOT}"
ssh_cmd "mkdir -p ${REMOTE_MOTUS}/pretrained_models ${REMOTE_ROOT}/runs ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/recipes"

echo "[2/6] rsync Motus 代码（排除 pretrained/outputs/.git）"
rsync_ssh \
  --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' \
  --exclude '.hf' --exclude '.modelscope' --exclude 'outputs' \
  --exclude 'pretrained_models' --exclude 'checkpoints' \
  "${LOCAL_MOTUS}/" "${HOST}:${REMOTE_MOTUS}/"

echo "[3/6] rsync recipes/motus 脚本"
rsync_ssh \
  "${HERE}/" "${HOST}:${REMOTE_ROOT}/recipes/"

echo "[4/6] rsync 预训练权重（Motus 15G + Wan2_2_pretrain 9.4G + Qwen3 4G = ~28G）"
if [[ -d "${LOCAL_MOTUS}/pretrained_models" ]]; then
  rsync_ssh \
    --exclude 'Wan2.2-TI2V-5B' \
    "${LOCAL_MOTUS}/pretrained_models/" "${HOST}:${REMOTE_MOTUS}/pretrained_models/"
else
  echo "WARN: local pretrained_models missing at ${LOCAL_MOTUS}/pretrained_models"
fi

echo "[5/6] rsync Wan2.2-TI2V-5B（32G，${SKIP_WAN22:+跳过}）"
if [[ "${SKIP_WAN22}" == "1" ]]; then
  echo "  SKIP_WAN22=1, 跳过 Wan2.2 上传"
  echo "  远端可自行下载: python -c 'from modelscope import snapshot_download; snapshot_download(\"Wan-AI/Wan2.2-TI2V-5B\", local_dir=\"${REMOTE_MOTUS}/pretrained_models/Wan2.2-TI2V-5B\")'"
elif [[ -d "${LOCAL_WAN22}" ]]; then
  rsync_ssh \
    --exclude '.cache' \
    "${LOCAL_WAN22}/" "${HOST}:${REMOTE_MOTUS}/pretrained_models/Wan2.2-TI2V-5B/"
else
  echo "WARN: Wan2.2 missing at ${LOCAL_WAN22}, remote will need to download"
fi

if [[ "${SKIP_ENV}" == "1" ]]; then
  echo "[6/6] SKIP_ENV=1, 跳过远端 conda 环境安装"
  echo "  手动运行: ssh ${HOST} && MOTUS_ROOT=${REMOTE_MOTUS} bash ${REMOTE_ROOT}/recipes/setup_env.sh"
else
  echo "[6/6] 远端 conda 环境安装 + 权重校验"
  # H100 = Hopper sm_90（不是 Blackwell sm_120）
  ssh_cmd "MOTUS_ROOT=${REMOTE_MOTUS} CONDA_ENV=motus TORCH_CUDA_ARCH_LIST=9.0 bash ${REMOTE_ROOT}/recipes/setup_env.sh"
  ssh_cmd "MOTUS_ROOT=${REMOTE_MOTUS} LOCAL_WAN22=${REMOTE_MOTUS}/pretrained_models/Wan2.2-TI2V-5B bash ${REMOTE_ROOT}/recipes/link_pretrained.sh"
fi

echo ""
echo "=========================================="
echo "  [OK] ${HOST} Motus 部署完成"
echo "=========================================="
echo ""
echo "  连接:  ssh -F ${SSH_CFG} ${HOST}"
echo "  激活:  source ${REMOTE_MOTUS}/env.sh"
echo "  训练:  bash ${REMOTE_ROOT}/recipes/launch_train.sh"
echo ""
echo "  远端目录结构:"
echo "    代码:     ${REMOTE_MOTUS}"
echo "    权重:     ${REMOTE_MOTUS}/pretrained_models/"
echo "    脚本:     ${REMOTE_ROOT}/recipes/"
echo "    训练输出: ${REMOTE_ROOT}/runs/"
echo ""
