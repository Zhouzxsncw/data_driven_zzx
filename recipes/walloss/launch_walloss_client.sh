#!/usr/bin/env bash
# 薄包装 → recipes/common/launch_client.sh
set -euo pipefail
unalias -a 2>/dev/null || true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
CONFIG="${WALLOSS_CONFIG:-$REPO_ROOT/infer/infer/config/lingbot/walloss_tongbot_ee_00091_local.yaml}"
exec bash "$REPO_ROOT/recipes/common/launch_client.sh" "$CONFIG" "$@"
