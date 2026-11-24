#!/bin/bash
# LingBot-VLA NPU 环境（在 lingbot-vla 容器内 source）。
ROOT="${LINGBOT_ROOT:-/home/lingbot-vla}"
SRC="${LINGBOT_SRC:-${ROOT}/src/lingbot-vla-v2}"
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
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-62200}"
export PYTHONPATH="${SRC}:${ROOT}/shim:${PYTHONPATH:-}"
export HF_HOME="${ROOT}/hf"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export LINGBOT_VLA_REPO="${SRC}"
export PRETRAINED_PATH="${ROOT}/pretrained/lingbot-vla-v2-6b"
export QWEN3VL_PATH="${ROOT}/pretrained/qwen3-tokenizer"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
if [ -x "${ROOT}/venv/bin/python" ]; then
  # shellcheck disable=SC1091
  source "${ROOT}/venv/bin/activate"
fi
echo "[env_npu] python=$(command -v python) NPUs=${ASCEND_RT_VISIBLE_DEVICES:-?} HCCL=${HCCL_IF_BASE_PORT}"
