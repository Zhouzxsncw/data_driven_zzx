#!/usr/bin/env bash
# [维护者] 联网下载 GR00T N1.5 权重与 tokenizer 到 data/hf_hub（不改 src/lerobot）
# 实习生使用仓库内 data/hf_hub；训练默认 GROOT_BOOTSTRAP_HF_CACHE=false。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
cd "$ROOT"
# shellcheck source=../train/scripts/lib/sim_offline_env.sh
source "${ROOT}/train/scripts/lib/sim_offline_env.sh"
sim_apply_offline_env
# shellcheck source=../train/scripts/lib/sim_train_common.sh
source "${ROOT}/train/scripts/lib/sim_train_common.sh"

PY="${LEROBOT_PYTHON:-${ROOT}/.venv310/bin/python}"
export PYTHONPATH="${ROOT}/train/src:${PYTHONPATH:-}"

GROOT_BASE="${GROOT_BASE_MODEL_PATH:-nvidia/GR00T-N1.5-3B}"
GROOT_TOK="${GROOT_TOKENIZER_ASSETS_REPO:-lerobot/eagle2hg-processor-groot-n1p5}"

# 下载阶段必须联网
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_ENDPOINT 2>/dev/null || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HUB_CACHE"
echo "HF_HOME=${HF_HOME}  HF_HUB_CACHE=${HF_HUB_CACHE}"

_download_repo() {
  local repo_id=$1
  echo "下载 GR00T: ${repo_id}"
  "$PY" -c "
from huggingface_hub import snapshot_download
import os
# 与 pi05/paligemma 一致：缓存到 \${HF_HOME}/hub/models--*
snapshot_download('${repo_id}')
print('OK ${repo_id}')
"
}

_download_repo "${GROOT_BASE}"
_download_repo "${GROOT_TOK}"

for _name in \
  "models--nvidia--GR00T-N1.5-3B" \
  "models--lerobot--eagle2hg-processor-groot-n1p5"; do
  if [[ ! -d "${HF_HUB_CACHE}/${_name}/snapshots" ]]; then
    echo "错误: 未找到 ${HF_HUB_CACHE}/${_name}" >&2
    exit 1
  fi
done
sim_sync_groot_eagle_processor_cache || {
  echo "错误: 无法同步 GR00T Eagle processor 到 data/lerobot/eagle2hg-processor-groot-n1p5" >&2
  exit 1
}
echo "GR00T 缓存就绪: ${HF_HUB_CACHE} + ${GROOT_EAGLE_PROCESSOR_DIR:-${HF_LEROBOT_HOME}/lerobot/eagle2hg-processor-groot-n1p5}"
