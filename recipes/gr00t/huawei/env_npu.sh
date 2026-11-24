#!/bin/bash
ROOT="${GR00T_ROOT:-/home/gr00t}"
SRC="${GR00T_SRC:-${ROOT}/src/Isaac-GR00T}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
# nnal/atb/set_env.sh 在部分 Atlas 镜像里会阻塞；训练不依赖它，跳过。
# source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

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
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-62400}"
export PYTHONPATH="${SRC}:${ROOT}/shim:${PYTHONPATH:-}"
export HF_HOME="${ROOT}/hf"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export GR00T_DIR="${GR00T_DIR:-${SRC}}"
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${ROOT}/pretrained/Cosmos-Reason2-2B}"
export GR00T_CKPT="${GR00T_CKPT:-${ROOT}/pretrained/GR00T-N1.7-3B}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export ROOT="${ROOT}"
export SRC="${SRC}"
export GR00T_ROOT="${ROOT}"
if [ -x "${ROOT}/venv/bin/python" ]; then
  # shellcheck disable=SC1091
  source "${ROOT}/venv/bin/activate"
fi
echo "[env_npu] python=$(command -v python) NPUs=${ASCEND_RT_VISIBLE_DEVICES:-?} HCCL=${HCCL_IF_BASE_PORT}"
