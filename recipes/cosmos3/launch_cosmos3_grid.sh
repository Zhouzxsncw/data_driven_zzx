#!/usr/bin/env bash
# 薄包装 → recipes/common/launch_client.sh
set -euo pipefail
unalias -a 2>/dev/null || true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
CONFIG="${COSMOS3_GRID_CONFIG:-$REPO_ROOT/infer/infer/config/lingbot/cosmos3_tongbot_16d_00073_grid.yaml}"
exec bash "$REPO_ROOT/recipes/common/launch_client.sh" "$CONFIG" "$@"
