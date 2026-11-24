#!/usr/bin/env bash
# [维护者] 联网下载 LIBERO 训练集与仿真资源并写入 data/；实习生直接使用已打包的 data/。
#
# 用法（仓库根）:
#   bash tools/setup_libero_offline_datasets.sh
#   bash tools/setup_libero_offline_datasets.sh --skip-combined   # 已有合并包时只拆分
#   bash tools/setup_libero_offline_datasets.sh --with-libero-90  # 尝试下载 libero_90（体积大）
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
cd "$ROOT"

# shellcheck source=../train/scripts/lib/sim_offline_env.sh
source "${ROOT}/train/scripts/lib/sim_offline_env.sh"
sim_apply_offline_env

PY="${LEROBOT_PYTHON:-${ROOT}/.venv310/bin/python3.10}"
HF="${ROOT}/.venv310/bin/hf"
export PYTHONPATH="${ROOT}/train/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
unset HF_HUB_ENDPOINT 2>/dev/null || true
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

LEROBOT_DIR="${OFFLINE_DATA_ROOT}/lerobot"
COMBINED="${LEROBOT_DIR}/libero"
LOG_DIR="${ROOT}/outputs/download_logs"
mkdir -p "$LOG_DIR" "$LEROBOT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/libero_offline_${TS}.log"

SKIP_COMBINED=0
WITH_LIBERO_90=0
for arg in "$@"; do
  case "$arg" in
    --skip-combined) SKIP_COMBINED=1 ;;
    --with-libero-90) WITH_LIBERO_90=1 ;;
  esac
done

dataset_complete() {
  [[ -f "$1/meta/info.json" ]]
}

_restore_hub_offline_flags() {
  local _prev_offline="$1"
  local _prev_tf="$2"
  [[ -n "${_prev_offline}" ]] && export HF_HUB_OFFLINE="${_prev_offline}"
  [[ -n "${_prev_tf}" ]] && export TRANSFORMERS_OFFLINE="${_prev_tf}"
}

download_repo_to() {
  local repo_id="$1"
  local dest="$2"
  # sim_apply_offline_env 可能开启 HF_HUB_OFFLINE；下载阶段必须联网
  local _prev_offline="${HF_HUB_OFFLINE:-}"
  local _prev_tf="${TRANSFORMERS_OFFLINE:-}"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

  if dataset_complete "${dest}"; then
    echo "[SKIP] ${repo_id} 已存在: ${dest}" | tee -a "$LOG"
    _restore_hub_offline_flags "${_prev_offline}" "${_prev_tf}"
    return 0
  fi
  mkdir -p "${dest}"
  echo "[DOWNLOAD] ${repo_id} -> ${dest}" | tee -a "$LOG"
  if "$HF" download "${repo_id}" --repo-type dataset --local-dir "${dest}" >>"$LOG" 2>&1; then
    if dataset_complete "${dest}"; then
      echo "[OK HF] ${repo_id}" | tee -a "$LOG"
      _restore_hub_offline_flags "${_prev_offline}" "${_prev_tf}"
      return 0
    fi
    # hf download 可能只拉了 data/ 未带 meta/（大仓库常见）
    if [[ ! -f "${dest}/meta/info.json" ]]; then
      echo "[HF] 补拉 meta/* -> ${dest}" | tee -a "$LOG"
      "$HF" download "${repo_id}" --repo-type dataset --local-dir "${dest}" \
        --include "meta/*" >>"$LOG" 2>&1 || true
    fi
    if dataset_complete "${dest}"; then
      echo "[OK HF+meta] ${repo_id}" | tee -a "$LOG"
      _restore_hub_offline_flags "${_prev_offline}" "${_prev_tf}"
      return 0
    fi
  fi
  echo "[LeRobotDataset] ${repo_id} ..." | tee -a "$LOG"
  if "$PY" - <<PY >>"$LOG" 2>&1
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset("${repo_id}", root="${dest}", download_videos=True)
print("OK", "${repo_id}")
PY
    dataset_complete "${dest}"
  then
    echo "[OK LeRobot] ${repo_id}" | tee -a "$LOG"
    _restore_hub_offline_flags "${_prev_offline}" "${_prev_tf}"
    return 0
  fi
  echo "[FAIL] ${repo_id}" | tee -a "$LOG"
  _restore_hub_offline_flags "${_prev_offline}" "${_prev_tf}"
  return 1
}

{
  echo "=== LIBERO offline setup $(date -Iseconds) ==="
  echo "OFFLINE_DATA_ROOT=${OFFLINE_DATA_ROOT}"
  echo "LEROBOT_DIR=${LEROBOT_DIR}"
} | tee -a "$LOG"

# --- 1) 合并包 spatial+object+goal+libero_10（Hub: lerobot/libero）---
if [[ "$SKIP_COMBINED" -eq 0 ]]; then
  download_repo_to "lerobot/libero" "${COMBINED}" || {
    echo "[WARN] lerobot/libero 下载失败，若已有缓存可: bash tools/dataset/bundle_offline_data.sh" | tee -a "$LOG"
  }
fi

if dataset_complete "${COMBINED}"; then
  echo "[INDEX] 生成各套件 episode 索引 -> ${LEROBOT_DIR}/_suite_indices/" | tee -a "$LOG"
  "$PY" "${ROOT}/tools/generate_libero_suite_episode_indices.py" \
    --combined-root "${COMBINED}" \
    --output-dir "${LEROBOT_DIR}/_suite_indices" 2>&1 | tee -a "$LOG"
else
  echo "[WARN] 无合并包 libero，spatial/object/goal 需独立目录或先下载 lerobot/libero" | tee -a "$LOG"
fi

# --- 2) libero_10：合并拆分可能已生成；否则单独拉取 ---
if ! dataset_complete "${LEROBOT_DIR}/libero_10"; then
  download_repo_to "lerobot/libero_10" "${LEROBOT_DIR}/libero_10" || {
    MS="${HOME}/.cache/modelscope/hub/datasets/lerobot/libero_10"
    if dataset_complete "${MS}"; then
      rm -rf "${LEROBOT_DIR}/libero_10"
      ln -sfn "$(readlink -f "${MS}")" "${LEROBOT_DIR}/libero_10"
      echo "[LINK] libero_10 <- ModelScope" | tee -a "$LOG"
    fi
  }
fi

# --- 3) libero_90（可选，官方 repo 常不可用）---
if [[ "$WITH_LIBERO_90" -eq 1 ]]; then
  if ! dataset_complete "${LEROBOT_DIR}/libero_90"; then
    echo "[INFO] libero_90: Hub 无 lerobot/libero_90，使用 jesbu1/libero_90_lerobot（约 70GB）" | tee -a "$LOG"
    download_repo_to "jesbu1/libero_90_lerobot" "${LEROBOT_DIR}/libero_90" || \
      download_repo_to "lerobot/libero_90" "${LEROBOT_DIR}/libero_90" || \
      echo "[WARN] libero_90 未就绪，见日志 ${LOG}" | tee -a "$LOG"
  fi
  if [[ -f "${LEROBOT_DIR}/libero_90/meta/info.json" ]] && \
     [[ "$("$PY" -c "import json;print(json.load(open('${LEROBOT_DIR}/libero_90/meta/info.json')).get('codebase_version',''))" 2>/dev/null || echo "")" != "v3.0" ]]; then
    echo "[REPAIR] libero_90 v2→v3 + 特征对齐" | tee -a "$LOG"
    "$PY" "${ROOT}/tools/dataset/repair_libero_90_dataset.py" \
      --data-root "${OFFLINE_DATA_ROOT}" \
      --skip-download \
      --allow-partial 2>&1 | tee -a "$LOG" || \
      echo "[WARN] repair_libero_90 失败，可手动: bash tools/dataset/fix_libero_90_dataset.sh" | tee -a "$LOG"
  fi
fi

# --- 4) LIBERO 仿真 assets + config ---
if [[ -f "${ROOT}/tools/bootstrap/bootstrap_libero_sim_assets.sh" ]]; then
  export LIBERO_ASSETS_DIR LIBERO_CONFIG_PATH
  ALLOW_LIBERO_ASSETS_DOWNLOAD=1 bash "${ROOT}/tools/bootstrap/bootstrap_libero_sim_assets.sh" >>"$LOG" 2>&1 || true
fi
sim_write_libero_config_if_missing || true

# --- 5) 各套件统计文件（训练用 quantile stats）---
if dataset_complete "${COMBINED}"; then
  "$PY" "${ROOT}/tools/dataset/ensure_dataset_quantile_stats.py" --root "${COMBINED}" >>"$LOG" 2>&1 || true
fi
for suite in libero_spatial libero_object libero_goal libero_10 libero_90; do
  d="${LEROBOT_DIR}/${suite}"
  if dataset_complete "${d}"; then
    "$PY" "${ROOT}/tools/dataset/ensure_dataset_quantile_stats.py" --root "${d}" >>"$LOG" 2>&1 || true
  fi
done

# --- 6) 清单 ---
MANIFEST="${OFFLINE_DATA_ROOT}/MANIFEST.txt"
{
  echo "# LIBERO offline — $(date -Iseconds)"
  echo "OFFLINE_DATA_ROOT=${OFFLINE_DATA_ROOT}"
  echo ""
  echo "## LIBERO LeRobot datasets (data/lerobot/)"
  if dataset_complete "${COMBINED}"; then
    du_h="$(du -shL "${COMBINED}" 2>/dev/null | cut -f1)"
    eps="$("$PY" -c "import json;print(json.load(open('${COMBINED}/meta/info.json')).get('total_episodes','?'))" 2>/dev/null || echo "?")"
    echo "  OK  libero (combined)  ${du_h}  episodes=${eps}"
  else
    echo "  --  libero (combined)  (missing)"
  fi
  for name in libero_spatial libero_object libero_goal libero_10; do
    idx="${LEROBOT_DIR}/_suite_indices/${name}.txt"
    if [[ -f "${idx}" ]]; then
      n="$(tr ',' '\n' <"${idx}" | grep -c . || echo 0)"
      echo "  OK  ${name}  (via libero + index)  episodes=${n}"
    elif dataset_complete "${LEROBOT_DIR}/${name}"; then
      d="${LEROBOT_DIR}/${name}"
      du_h="$(du -shL "${d}" 2>/dev/null | cut -f1)"
      eps="$("$PY" -c "import json;print(json.load(open('${d}/meta/info.json')).get('total_episodes','?'))" 2>/dev/null || echo "?")"
      echo "  OK  ${name}  ${du_h}  episodes=${eps}"
    else
      echo "  --  ${name}  (missing)"
    fi
  done
  d="${LEROBOT_DIR}/libero_90"
  if dataset_complete "${d}"; then
    du_h="$(du -shL "${d}" 2>/dev/null | cut -f1)"
    eps="$("$PY" -c "import json;print(json.load(open('${d}/meta/info.json')).get('total_episodes','?'))" 2>/dev/null || echo "?")"
    echo "  OK  libero_90  ${du_h}  episodes=${eps}"
  else
    echo "  --  libero_90  (missing; 需 --with-libero-90)"
  fi
  echo ""
  echo "## LIBERO sim assets"
  if [[ -d "${LIBERO_ASSETS_DIR}/scenes" ]]; then
    echo "  OK  $(du -sh "${LIBERO_ASSETS_DIR}" 2>/dev/null | cut -f1)"
  else
    echo "  MISSING"
  fi
} | tee "$MANIFEST"

echo ""
echo "完成。训练/评估默认使用: OFFLINE_DATA_ROOT=${OFFLINE_DATA_ROOT}"
echo "  SIM_ENV=libero  SIM_ENV_TASK=<套件名>  ->  data/lerobot/<套件名>/"
echo "日志: ${LOG}"
echo "清单: ${MANIFEST}"
