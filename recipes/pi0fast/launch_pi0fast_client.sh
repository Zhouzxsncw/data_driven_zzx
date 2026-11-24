#!/usr/bin/env bash
# 薄包装 → recipes/common/launch_client.sh
set -euo pipefail
unalias -a 2>/dev/null || true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
CONFIG="${PI0FAST_CONFIG:-$REPO_ROOT/infer/infer/config/lerobot/pi0fast_openpi_tongbot_16d_00036_ws_client.yaml}"
if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] 找不到配置: $CONFIG" >&2
  echo "        先跑: python3 $ROOT/gen_client_config.py" >&2
  exit 1
fi
exec bash "$REPO_ROOT/recipes/common/launch_client.sh" "$CONFIG" "$@"
