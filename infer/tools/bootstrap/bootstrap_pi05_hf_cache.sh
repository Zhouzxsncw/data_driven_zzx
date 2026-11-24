#!/usr/bin/env bash
# [维护者] 将 paligemma tokenizer 放入 data/hf_hub（或链入 ModelScope 缓存）
# 实习生默认使用仓库内 data/hf_hub，不调用本脚本。
# 用法: ALLOW_PI05_BOOTSTRAP_DOWNLOAD=1 bash tools/bootstrap/bootstrap_pi05_hf_cache.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
if [[ -f "${ROOT}/train/scripts/lib/sim_offline_env.sh" ]]; then
  # shellcheck source=../train/scripts/lib/sim_offline_env.sh
  source "${ROOT}/train/scripts/lib/sim_offline_env.sh"
  sim_apply_offline_env
fi

PY="${LEROBOT_PYTHON:-python3}"
MS_PALIGEMMA="${MODELSCOPE_PALIGEMMA_DIR:-${HOME}/.cache/modelscope/hub/models/AI-ModelScope/paligemma-3b-pt-224}"
HF_PALIGEMMA="${HF_HOME}/hub/models--google--paligemma-3b-pt-224"

if declare -f sim_paligemma_tokenizer_local >/dev/null 2>&1 && sim_paligemma_tokenizer_local; then
  echo "bootstrap: paligemma tokenizer 已在 ${HF_HOME}"
  exit 0
fi

if [[ "${ALLOW_PI05_BOOTSTRAP_DOWNLOAD:-}" != "1" ]]; then
  echo "错误: data/hf_hub 中无 paligemma tokenizer" >&2
  echo "  实习生: 使用含完整 data/hf_hub/ 的仓库（见 data/README.md）" >&2
  echo "  维护者: ALLOW_PI05_BOOTSTRAP_DOWNLOAD=1 bash $0" >&2
  exit 1
fi

_link_ms_to_hf() {
  local ms_dir=$1 hf_repo_dir=$2 snap_name=${3:-modelscope-link}
  if [[ ! -d "$ms_dir" ]]; then
    echo "bootstrap: 跳过（不存在）$ms_dir" >&2
    return 1
  fi
  mkdir -p "${hf_repo_dir}/snapshots" "${hf_repo_dir}/refs"
  rm -rf "${hf_repo_dir}/snapshots/${snap_name}"
  cp -al "$ms_dir" "${hf_repo_dir}/snapshots/${snap_name}" 2>/dev/null \
    || cp -a "$ms_dir" "${hf_repo_dir}/snapshots/${snap_name}"
  echo -n "$snap_name" > "${hf_repo_dir}/refs/main"
  echo "bootstrap: HF 缓存已链接 ${hf_repo_dir} <- ${ms_dir}"
}

if [[ ! -d "$MS_PALIGEMMA" ]]; then
  echo "bootstrap: 下载 paligemma tokenizer（ModelScope）..." >&2
  "$PY" -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/paligemma-3b-pt-224', allow_file_pattern=['tokenizer*','*.json','spiece.model','tokenizer.model','special_tokens_map*','added_tokens.json','preprocessor_config.json'])"
fi
_link_ms_to_hf "$MS_PALIGEMMA" "$HF_PALIGEMMA"

"$PY" -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224'); print('bootstrap: paligemma tokenizer OK')"
