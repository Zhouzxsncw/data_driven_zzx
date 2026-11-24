#!/usr/bin/env bash
# 修复 data/lerobot/libero_90 至完整 v3.0（约 3959 episodes），并 verify + smoke train/eval。
# 不修改 src/lerobot；逻辑在 tools/dataset/repair_libero_90_dataset.py。
#
# 用法（仓库根）:
#   bash tools/dataset/fix_libero_90_dataset.sh              # 全量：staging 下载 + 转换 + 替换
#   bash tools/dataset/fix_libero_90_dataset.sh --smoke-only # 仅验证已有子集（不等待全量）
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
cd "$ROOT"
# shellcheck source=../train/scripts/lib/sim_offline_env.sh
source "${ROOT}/train/scripts/lib/sim_offline_env.sh"
sim_apply_offline_env

PY="${LEROBOT_PYTHON:-${ROOT}/.venv310/bin/python3.10}"
export PYTHONPATH="${ROOT}/train/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
unset HF_HUB_ENDPOINT HF_HUB_OFFLINE TRANSFORMERS_OFFLINE 2>/dev/null || true
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

LOG_DIR="${ROOT}/outputs/download_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/libero_90_fix_${TS}.log"

DEST="${OFFLINE_DATA_ROOT}/lerobot/libero_90"
BUILD="${OFFLINE_DATA_ROOT}/lerobot/.libero_90_build"
EXPECTED=3959

SMOKE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --smoke-only) SMOKE_ONLY=1 ;;
  esac
done

exec > >(tee -a "$LOG") 2>&1
echo "=== libero_90 修复日志 $(date -Iseconds) ==="
echo "LOG=$LOG"
echo "DEST=$DEST"
echo "BUILD=$BUILD"

if [[ "$SMOKE_ONLY" -eq 1 ]]; then
  echo "=== smoke 模式：修复当前 DEST（不强制 ${EXPECTED} episodes）==="
  "$PY" "${ROOT}/tools/dataset/repair_libero_90_dataset.py" \
    --root "$DEST" --skip-download --allow-partial --max-episodes=32 --validate
else
  # 若已是完整 v3，跳过重建
  if "$PY" "${ROOT}/tools/dataset/repair_libero_90_dataset.py" --root "$DEST" --check-only 2>/dev/null; then
    echo "[SKIP] ${DEST} 已是完整 v3.0（>= ${EXPECTED} episodes）"
  else
    if [[ -d "$DEST" ]]; then
      PARTIAL="${OFFLINE_DATA_ROOT}/lerobot/libero_90.partial.${TS}"
      echo "[BACKUP] 不完整/旧版数据集 -> ${PARTIAL}"
      mv "$DEST" "$PARTIAL"
    fi
    if [[ -d "$BUILD" ]] && [[ "$(find "$BUILD/data" -name 'episode_*.parquet' 2>/dev/null | wc -l)" -gt 0 ]]; then
      echo "[RESUME] 继续 staging 下载/修复: $BUILD"
    else
      rm -rf "$BUILD"
      mkdir -p "$BUILD"
    fi

    echo "=== 1/5 全量下载 v2 原始数据到 staging（约 70GB，可续传）==="
    "$PY" "${ROOT}/tools/dataset/repair_libero_90_dataset.py" --root "$BUILD"

    echo "=== 2/5 安装到 ${DEST} ==="
    mv "$BUILD" "$DEST"

    echo "=== 3/5 最终校验 ==="
    "$PY" "${ROOT}/tools/dataset/repair_libero_90_dataset.py" --root "$DEST" --check-only
    "$PY" "${ROOT}/tools/dataset/repair_libero_90_dataset.py" --root "$DEST" --validate-only
  fi
fi

echo "=== 4/5 quantile stats + 套件预检 ==="
if [[ -f "${DEST}/meta/stats.json" ]]; then
  "$PY" "${ROOT}/tools/dataset/ensure_dataset_quantile_stats.py" --root "$DEST"
fi
bash "${ROOT}/tools/dataset/verify_libero_suite_datasets.sh"

echo "=== 5/5 libero_90 smoke 训练 + eval (ACT) ==="
export SIM_ENV=libero
export SIM_ENV_TASK=libero_90
export STEPS=200
export EVAL_FREQ=200
export SAVE_FREQ=999999
export BATCH_SIZE=1
export NUM_WORKERS=0
export SIM_EVAL_N_EPISODES=2
export SIM_EVAL_BATCH_SIZE=1
export ONLY_SUITE=libero_90
export ONLY_POLICY=ACT
export CONTINUE_ON_FAIL=false
bash "${ROOT}/train/scripts/test/test_libero_all_suites_all_policies_smoke.sh"

# 更新 MANIFEST 条目
MANIFEST="${OFFLINE_DATA_ROOT}/MANIFEST.txt"
if [[ -f "${DEST}/meta/info.json" ]]; then
  EPS="$("$PY" -c "import json;print(json.load(open('${DEST}/meta/info.json')).get('total_episodes','?'))")"
  DU="$(du -shL "${DEST}" 2>/dev/null | cut -f1)"
  {
    echo ""
    echo "# libero_90 fix ${TS}"
    echo "  OK  libero_90  ${DU}  episodes=${EPS}"
  } >>"$MANIFEST"
fi

echo ""
echo "完成。libero_90: ${DEST}"
echo "日志: ${LOG}"
