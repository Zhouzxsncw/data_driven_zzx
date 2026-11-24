#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/recipes/pi05/host_env.sh"
CONFIG="$INFER_ROOT/infer/config/lerobot/pi05_current_tk_ws.yaml"
cd "$INFER_ROOT"
export PYTHONPATH="$INFER_ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1
if [[ "${1:-}" == --help ]]; then
  exec "$PI05_PYTHON" -B -m infer.cli.run_pico_teleop --help
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: bash run_pico_teleop_zzx.sh" >&2
  exit 2
fi
exec "$PI05_PYTHON" -B -m infer.cli.tk_pico_session --root "$ROOT" --config "$CONFIG" --timeout "${PICO_READY_TIMEOUT:-90}" -- "$PI05_PYTHON" -B -m infer.cli.run_pico_teleop --config "$CONFIG"
