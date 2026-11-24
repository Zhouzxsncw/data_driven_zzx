#!/usr/bin/env bash
# 预检：LIBERO 五套件 × sim_train_common 数据集解析（不启动训练）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
cd "$ROOT"
# shellcheck source=../train/scripts/lib/sim_train_common.sh
source "${ROOT}/train/scripts/lib/sim_train_common.sh"

SUITES=(libero_spatial libero_object libero_goal libero_10 libero_90)
export SIM_ENV=libero

fail=0
echo "LIBERO 套件数据集预检 (OFFLINE_DATA_ROOT=${OFFLINE_DATA_ROOT:-${ROOT}/data})"
echo "================================================================"

for suite in "${SUITES[@]}"; do
  export SIM_ENV_TASK="$suite"
  unset DATASET_REPO_ID DATASET_ROOT DATASET_EPISODES
  sim_resolve_dataset_paths
  ok=1
  if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    ok=0
  fi
  if [[ -n "${DATASET_EPISODES:-}" ]]; then
    idx="${OFFLINE_DATA_ROOT:-${ROOT}/data}/lerobot/_suite_indices/${suite}.txt"
    [[ -f "$idx" ]] || ok=0
  fi
  ep_count=""
  if [[ -n "${DATASET_EPISODES:-}" ]]; then
    ep_count=$(echo "$DATASET_EPISODES" | tr ',' '\n' | grep -c . || true)
  fi
  if [[ "$ok" -eq 1 ]]; then
    echo "OK   ${suite}"
    echo "     repo=${DATASET_REPO_ID} root=${DATASET_ROOT}"
    [[ -n "${DATASET_EPISODES:-}" ]] && echo "     episodes=${ep_count} (combined+index)"
  else
    echo "FAIL ${suite}"
    echo "     repo=${DATASET_REPO_ID:-?} root=${DATASET_ROOT:-?}"
    [[ -n "${DATASET_EPISODES:-}" ]] && echo "     episodes subset requested but data incomplete"
    fail=$((fail + 1))
  fi
done

echo "================================================================"
if [[ "$fail" -gt 0 ]]; then
  echo "失败: ${fail} 个套件数据未就绪"
  echo "修复: 使用含 data/lerobot/ 与 _suite_indices/ 的完整 data/ 包（见 data/README.md）"
  echo "维护者填充: bash tools/setup_libero_offline_datasets.sh [--with-libero-90]"
  exit 1
fi
echo "全部套件数据路径解析通过"
