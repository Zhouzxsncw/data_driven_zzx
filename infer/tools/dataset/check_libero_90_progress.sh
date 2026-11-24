#!/usr/bin/env bash
# 查看 libero_90 全量修复进度（下载 / stats / convert）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
BUILD="${ROOT}/data/lerobot/.libero_90_build"
DEST="${ROOT}/data/lerobot/libero_90"
LOG="${1:-/tmp/libero_90_full_fix.log}"

echo "=== libero_90 修复进度 $(date -Iseconds) ==="
if pgrep -af "fix_libero_90_dataset|repair_libero_90" >/dev/null 2>&1; then
  echo "进程: 运行中"
  pgrep -af "fix_libero_90|repair_libero" | head -3
else
  echo "进程: 未运行"
fi

for d in "$BUILD" "$DEST"; do
  [[ -d "$d" ]] || continue
  n_ep=$(find "$d/data" -name "episode_*.parquet" 2>/dev/null | wc -l)
  n_file=$(find "$d/data" -name "file-*.parquet" 2>/dev/null | wc -l)
  du_h=$(du -sh "$d" 2>/dev/null | cut -f1)
  if [[ -f "$d/meta/info.json" ]]; then
    ver=$(python3 -c "import json;print(json.load(open('$d/meta/info.json')).get('codebase_version','?'), json.load(open('$d/meta/info.json')).get('total_episodes','?'))" 2>/dev/null || echo "? ?")
    echo "$d: ${du_h}  episode_parquet=${n_ep}  v3_files=${n_file}  meta=${ver}"
  else
    echo "$d: ${du_h}  episode_parquet=${n_ep}  v3_files=${n_file}  (no meta)"
  fi
done

if [[ -f "$LOG" ]]; then
  echo "--- log tail ---"
  tail -5 "$LOG"
fi
