#!/usr/bin/env bash
# [维护者] 联网下载 MetaWorld + LIBERO LeRobot 数据集，再执行 bundle_offline_data.sh 打入 data/
# 实习生训练/评估请使用仓库内 data/lerobot/，勿运行本脚本。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
PY="${LEROBOT_PYTHON:-${ROOT}/.venv310/bin/python3.10}"
HF="${ROOT}/.venv310/bin/hf"

export PYTHONPATH="${ROOT}/train/src:${PYTHONPATH:-}"
if [[ -f "${ROOT}/train/scripts/lib/sim_offline_env.sh" ]]; then
  # shellcheck source=../train/scripts/lib/sim_offline_env.sh
  source "${ROOT}/train/scripts/lib/sim_offline_env.sh"
  sim_apply_offline_env
else
  export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
  export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
fi
# 数据集落盘: data/lerobot/<name>（与 sim_train_common 一致）
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${ROOT}/data}"
# 强制官方 Hub（镜像站缺少 metaworld / 多数 libero 仓库）
export HF_ENDPOINT="https://huggingface.co"
unset HF_HUB_ENDPOINT 2>/dev/null || true
# 大文件走常规 CDN，避免 cas-bridge.xethub 超时；拉长读超时便于断点续传
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

LOG_DIR="${ROOT}/outputs/download_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/metaworld_libero_${TS}.log"

DATASETS=(
  "lerobot/metaworld_mt50"
  "lerobot/libero"
  "lerobot/libero_10"
  "lerobot/libero_90"
)

dataset_complete() {
  local dest="$1"
  [[ -f "${dest}/meta/info.json" ]]
}

download_via_modelscope() {
  local repo_id="$1"
  local dest="$2"
  echo "[ModelScope] ${repo_id} ..." | tee -a "$LOG"
  "$PY" - <<PY >>"$LOG" 2>&1
from modelscope import snapshot_download
path = snapshot_download("${repo_id}", repo_type="dataset")
print("ModelScope path:", path)
PY
  local ms_path="${HOME}/.cache/modelscope/hub/datasets/${repo_id}"
  if [[ ! -f "${ms_path}/meta/info.json" ]]; then
    echo "[ModelScope FAIL] 未找到 ${ms_path}/meta/info.json" | tee -a "$LOG"
    return 1
  fi
  rm -rf "${dest}"
  ln -sfn "$(readlink -f "${ms_path}")" "${dest}"
  echo "[OK ModelScope] ${repo_id} -> ${dest}" | tee -a "$LOG"
}

download_repo() {
  local repo_id="$1"
  local name="${repo_id#lerobot/}"
  local dest="${HF_LEROBOT_HOME}/lerobot/${name}"
  local _prev_offline="${HF_HUB_OFFLINE:-}"
  local _prev_tf="${TRANSFORMERS_OFFLINE:-}"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

  if dataset_complete "${dest}"; then
    echo "[SKIP] ${repo_id} 已存在: ${dest}" | tee -a "$LOG"
    return 0
  fi

  # 清掉之前失败留下的空目录，便于 hf download 断点续传
  if [[ -d "${dest}" && ! -f "${dest}/meta/info.json" ]]; then
    local incomplete=0
    if [[ -L "${dest}" ]]; then
      rm -f "${dest}"
    elif find "${dest}" -maxdepth 3 -type f ! -name 'README.md' 2>/dev/null | grep -q .; then
      incomplete=1
      echo "[RESUME] ${repo_id} 检测到未完成下载: ${dest}" | tee -a "$LOG"
    else
      echo "[CLEAN] 移除空目录 ${dest}" | tee -a "$LOG"
      rm -rf "${dest}"
    fi
    if [[ "${incomplete}" -eq 0 ]]; then
      mkdir -p "${dest}"
    fi
  else
    mkdir -p "${dest}"
  fi

  echo "[START HF] ${repo_id} -> ${dest}" | tee -a "$LOG"

  if "$HF" download "${repo_id}" --repo-type dataset --local-dir "${dest}" >>"$LOG" 2>&1; then
    if dataset_complete "${dest}"; then
      echo "[OK HF] ${repo_id}" | tee -a "$LOG"
      return 0
    fi
    echo "[HF INCOMPLETE] ${repo_id} 缺少 meta/info.json" | tee -a "$LOG"
  else
    echo "[HF CLI FAIL] ${repo_id}" | tee -a "$LOG"
  fi

  if [[ "${repo_id}" == "lerobot/libero_10" ]]; then
    download_via_modelscope "${repo_id}" "${dest}" && return 0
  fi

  echo "[LeRobotDataset] 重试 ${repo_id} ..." | tee -a "$LOG"
  if "$PY" - <<PY >>"$LOG" 2>&1
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset("${repo_id}", root="${dest}", download_videos=True)
print("LeRobotDataset OK: ${repo_id}")
PY
  then
    if dataset_complete "${dest}"; then
      echo "[OK LeRobot] ${repo_id}" | tee -a "$LOG"
      return 0
    fi
  fi

  [[ -n "${_prev_offline}" ]] && export HF_HUB_OFFLINE="${_prev_offline}"
  [[ -n "${_prev_tf}" ]] && export TRANSFORMERS_OFFLINE="${_prev_tf}"
  return 1
}

{
  echo "=== download start $(date -Iseconds) ==="
  echo "HF_ENDPOINT=${HF_ENDPOINT}"
  echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
} | tee -a "$LOG"

for repo in "${DATASETS[@]}"; do
  download_repo "$repo" || echo "[WARN] ${repo} 下载失败，见日志" | tee -a "$LOG"
done

# spatial / object / goal：由合并包生成 episode 索引（训练时过滤子集）
if [[ -f "${HF_LEROBOT_HOME}/lerobot/libero/meta/info.json" ]]; then
  echo "[INDEX] lerobot/libero -> _suite_indices/" | tee -a "$LOG"
  "$PY" "${ROOT}/tools/generate_libero_suite_episode_indices.py" \
    --combined-root "${HF_LEROBOT_HOME}/lerobot/libero" \
    --output-dir "${HF_LEROBOT_HOME}/lerobot/_suite_indices" >>"$LOG" 2>&1 \
    || echo "[WARN] LIBERO 索引生成失败" | tee -a "$LOG"
else
  echo "[WARN] 无 lerobot/libero，跳过套件索引" | tee -a "$LOG"
  echo "  可单独执行: bash tools/setup_libero_offline_datasets.sh" | tee -a "$LOG"
fi

# 训练脚本默认 lerobot/metaworld；若仅 mt50 成功则做符号链接
MW="${HF_LEROBOT_HOME}/lerobot/metaworld"
MT50="${HF_LEROBOT_HOME}/lerobot/metaworld_mt50"
if [[ ! -f "${MW}/meta/info.json" && -f "${MT50}/meta/info.json" ]]; then
  echo "[LINK] metaworld -> metaworld_mt50" | tee -a "$LOG"
  ln -sfn "$(readlink -f "${MT50}")" "${MW}"
fi

echo "=== download end $(date -Iseconds) ===" | tee -a "$LOG"
echo "日志: ${LOG}"
