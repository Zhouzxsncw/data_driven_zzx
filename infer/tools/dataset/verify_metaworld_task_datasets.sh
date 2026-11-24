#!/usr/bin/env bash
# 校验 MetaWorld 合并集 + 各单任务 episode 索引（_metaworld_indices/<task>.txt）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
cd "$ROOT"

DATA_ROOT="${OFFLINE_DATA_ROOT:-${ROOT}/data}"
MW_ROOT="${DATA_ROOT}/lerobot/metaworld"
IDX_DIR="${DATA_ROOT}/lerobot/_metaworld_indices"
CFG="${ROOT}/train/src/lerobot/envs/metaworld_config.json"

fail=0

if [[ ! -f "${MW_ROOT}/meta/info.json" ]]; then
  echo "FAIL: 缺少 ${MW_ROOT}/meta/info.json"
  echo "  请使用含 data/lerobot/metaworld/ 的本地 data/ 包（见 data/README.md）"
  exit 1
fi
echo "OK: metaworld root ${MW_ROOT}"

if [[ ! -f "$CFG" ]]; then
  echo "FAIL: 缺少 ${CFG}"
  exit 1
fi

"${LEROBOT_PYTHON:-${ROOT}/.venv310/bin/python}" tools/dataset/repair_metaworld_dataset.py \
  --metaworld-root "$MW_ROOT" \
  --fix-push-collision \
  --regenerate-indices \
  >/dev/null 2>&1 || true

if [[ ! -d "$IDX_DIR" ]] || [[ -z "$(ls -A "$IDX_DIR"/*.txt 2>/dev/null)" ]]; then
  echo "WARN: 未生成任务索引，正在运行 generate_metaworld_task_episode_indices.py ..."
  "${LEROBOT_PYTHON:-${ROOT}/.venv310/bin/python}" tools/dataset/generate_metaworld_task_episode_indices.py \
    --metaworld-root "$MW_ROOT" \
    --output-dir "$IDX_DIR"
fi

while IFS= read -r task_key; do
  idx="${IDX_DIR}/${task_key}.txt"
  if [[ ! -f "$idx" ]]; then
    echo "FAIL: 缺少索引 ${idx}"
    fail=1
    continue
  fi
  n=$(tr ',' '\n' <"$idx" | grep -c . || true)
  if [[ "$n" -lt 1 ]]; then
    echo "FAIL: ${task_key} 索引为空"
    fail=1
  elif [[ "$n" -ne 50 ]]; then
    echo "WARN: ${task_key} episodes=${n} (期望 50)"
  else
    echo "OK: ${task_key} episodes=${n}"
  fi
done < <("${LEROBOT_PYTHON:-${ROOT}/.venv310/bin/python}" -c "
import json
from pathlib import Path
cfg = json.loads(Path('${CFG}').read_text())
for k in sorted(cfg['TASK_NAME_TO_ID'], key=cfg['TASK_NAME_TO_ID'].get):
    print(k)
")

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "MetaWorld 任务索引预检通过 (${IDX_DIR})"
