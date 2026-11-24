#!/bin/bash
# WALL-OSS NPU 环境（在 wall-oss 容器内 source）。
# 路径对齐本机 /data/wall-oss → /home/wall-oss。
ROOT="${WALLOSS_ROOT:-/home/wall-oss}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  n=0
  for i in $(seq 0 15); do
    [ -e "/dev/davinci${i}" ] && n=$((i + 1))
  done
  if [ "$n" -gt 0 ]; then
    seq_list=$(seq -s, 0 $((n - 1)))
    export ASCEND_RT_VISIBLE_DEVICES="$seq_list"
  fi
fi
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-61200}"
export PYTHONPATH="${ROOT}/src/wall-x:${ROOT}/src/lerobot:${ROOT}/shim:${PYTHONPATH:-}"
export HF_HOME="${ROOT}/hf"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export PRETRAINED_PATH="${ROOT}/pretrained/wall-oss-0.5"
export PROCESSOR_PATH="${ROOT}/pretrained/Qwen2.5-VL-3B-Instruct"
export WALLX_CODE="${ROOT}/src/wall-x"
export WALLX_SKIP_CUDA_EXT=1
export TORCHDYNAMO_DISABLE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [ -x "${ROOT}/venv/bin/python" ]; then
  # shellcheck disable=SC1091
  source "${ROOT}/venv/bin/activate"
fi
echo "[env_npu] python=$(command -v python) NPUs=${ASCEND_RT_VISIBLE_DEVICES:-?} HCCL=${HCCL_IF_BASE_PORT}"
