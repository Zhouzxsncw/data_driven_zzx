#!/usr/bin/env bash
# RoboTwin 2.0 客户端。仿真 Python 用 $SIM_ROOT/envs/robotwin。
#
#   source "$SIM_ROOT/env.sh" robotwin
#   PI05_EMBODIMENT=robotwin bash recipes/pi05/launch_pi05_server.sh
#   bash recipes/pi05/launch_robotwin_client.sh
set -euo pipefail
STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$STAGING/../.." && pwd)"
INFER_ROOT="${INFER_ROOT:-${REPO_ROOT}/infer}"
PYTHON="${ROBOTWIN_PYTHON:-${SIM_ROOT:-}/envs/robotwin/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "[ERROR] 没有 RoboTwin Python：$PYTHON 。先设 SIM_ROOT 并跑 scripts/bootstrap_sim_infer.sh" >&2
  exit 1
fi
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${SIM_ROOT:-}/src/RoboTwin}"
export PYTHONPATH="${INFER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
exec "$PYTHON" -u "$STAGING/eval_robotwin_client.py" "$@"
