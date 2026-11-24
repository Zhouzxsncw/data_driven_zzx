#!/usr/bin/env bash
# [维护者] 联网下载 ALOHA 仿真数据集并写入 data/lerobot/；实习生使用已打包的 data/。
#
# 用法（仓库根）:
#   bash tools/setup_aloha_offline_datasets.sh
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
LOG_DIR="${ROOT}/outputs/download_logs"
mkdir -p "$LOG_DIR" "$LEROBOT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/aloha_offline_${TS}.log"

# LeRobot 注册的 8 个 ALOHA 仿真数据集（见 train/src/lerobot/__init__.py）
ALOHA_DATASETS=(
  "lerobot/aloha_sim_insertion_human"
  "lerobot/aloha_sim_insertion_scripted"
  "lerobot/aloha_sim_transfer_cube_human"
  "lerobot/aloha_sim_transfer_cube_scripted"
  "lerobot/aloha_sim_insertion_human_image"
  "lerobot/aloha_sim_insertion_scripted_image"
  "lerobot/aloha_sim_transfer_cube_human_image"
  "lerobot/aloha_sim_transfer_cube_scripted_image"
)

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
  if "$HF" download "${repo_id}" --repo-type dataset --local-dir "${dest}" >>"$LOG" 2>&1 \
    && dataset_complete "${dest}"; then
    echo "[OK HF] ${repo_id}" | tee -a "$LOG"
    _restore_hub_offline_flags "${_prev_offline}" "${_prev_tf}"
    return 0
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
  echo "=== ALOHA offline setup $(date -Iseconds) ==="
  echo "LEROBOT_DIR=${LEROBOT_DIR}"
} | tee -a "$LOG"

FAIL=0
for repo in "${ALOHA_DATASETS[@]}"; do
  name="${repo#lerobot/}"
  dest="${LEROBOT_DIR}/${name}"
  download_repo_to "${repo}" "${dest}" || FAIL=$((FAIL + 1))
done

# 训练默认路径：aloha_sim -> insertion_human（与 AlohaInsertion-v0 默认任务对齐）
if dataset_complete "${LEROBOT_DIR}/aloha_sim_insertion_human" \
  && ! dataset_complete "${LEROBOT_DIR}/aloha_sim"; then
  echo "[LINK] aloha_sim -> aloha_sim_insertion_human" | tee -a "$LOG"
  ln -sfn "aloha_sim_insertion_human" "${LEROBOT_DIR}/aloha_sim"
fi

for name in aloha_sim_insertion_human aloha_sim_insertion_scripted \
  aloha_sim_transfer_cube_human aloha_sim_transfer_cube_scripted \
  aloha_sim_insertion_human_image aloha_sim_insertion_scripted_image \
  aloha_sim_transfer_cube_human_image aloha_sim_transfer_cube_scripted_image; do
  d="${LEROBOT_DIR}/${name}"
  if dataset_complete "${d}"; then
    "$PY" "${ROOT}/tools/dataset/ensure_dataset_quantile_stats.py" --root "${d}" >>"$LOG" 2>&1 || true
  fi
done

MANIFEST="${OFFLINE_DATA_ROOT}/MANIFEST.txt"
{
  echo ""
  echo "## ALOHA sim datasets (appended $(date -Iseconds))"
  for repo in "${ALOHA_DATASETS[@]}"; do
    name="${repo#lerobot/}"
    d="${LEROBOT_DIR}/${name}"
    if dataset_complete "${d}"; then
      du_h="$(du -shL "${d}" 2>/dev/null | cut -f1)"
      eps="$("$PY" -c "import json;print(json.load(open('${d}/meta/info.json')).get('total_episodes','?'))" 2>/dev/null || echo "?")"
      echo "  OK  ${name}  ${du_h}  episodes=${eps}"
    else
      echo "  --  ${name}  (missing)"
    fi
  done
} | tee -a "$MANIFEST"

echo ""
echo "完成。Aloha 训练示例:"
echo "  SIM_ENV=aloha SIM_ENV_TASK=AlohaInsertion-v0  -> data/lerobot/aloha_sim_insertion_human"
echo "  SIM_ENV=aloha SIM_ENV_TASK=AlohaTransferCube-v0 -> data/lerobot/aloha_sim_transfer_cube_human"
echo "日志: ${LOG}"
exit "${FAIL}"
