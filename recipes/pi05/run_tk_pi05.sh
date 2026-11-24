#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -lt 3 || ! "$1" =~ ^(check|server|client)$ ]]; then
  echo "Usage: bash $0 check|server|client /absolute/path/config.yaml /absolute/path/pretrained_model" >&2
  exit 2
fi
MODE="$1"
export PI05_CONFIG="$2" PI05_CKPT="$3"
source "$ROOT/recipes/pi05/host_env.sh"
[[ -f "$PI05_CONFIG" ]] || { echo "Missing YAML: $PI05_CONFIG"; exit 1; }
[[ -f "$PI05_CKPT/model.safetensors" || -d "$PI05_CKPT/params" ]] || { echo "Missing weights: $PI05_CKPT"; exit 1; }
[[ -f "${PI05_CKPT%/pretrained_model}/assets/norm_stats.json" ]] || { echo "Missing sibling assets/norm_stats.json"; exit 1; }
cd "$INFER_ROOT"
export PYTHONPATH="$INFER_ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1
if [[ "$MODE" == check ]]; then
  "$PI05_PYTHON" -c 'import json,os,torch; from infer.policies.pi05_embodiment import launch_meta; from infer.policies.openpi_pi05_runtime import OpenpiPi05Runtime; print(json.dumps(launch_meta(os.environ["PI05_CONFIG"]),ensure_ascii=False,indent=2)); print("CUDA",torch.cuda.is_available()); print("PATH_AND_IMPORT_CHECK_OK (weights not loaded)")'
elif [[ "$MODE" == client ]]; then
  if pgrep -f -- "(^|[[:space:]])-m[[:space:]]+infer\.cli\.run_(async_)?infer([[:space:]]|$)" >/dev/null; then
    echo "Inference client already active. Shared RM75 controls must have one owner." >&2
    exit 1
  fi
  echo "=== OpenPI π0.5 异步动作块客户端 ==="
  echo "Python: $PI05_PYTHON"
  echo "Config: $PI05_CONFIG"
  echo "Entry:  python -m infer.cli.run_async_infer"
  echo "=================================="
  client=("$PI05_PYTHON" -m infer.cli.run_async_infer --config "$PI05_CONFIG" "${@:4}")
  for arg in "${@:4}"; do
    if [[ "$arg" == --pico-recovery ]]; then
      exec "$PI05_PYTHON" -B -m infer.cli.tk_pico_session --root "$ROOT" --config "$PI05_CONFIG"         --timeout "${PICO_READY_TIMEOUT:-90}" -- "${client[@]}"
    fi
  done
  exec flock -n "$ROOT/.rm75-control.lock" "${client[@]}"
else
  exec bash "$ROOT/recipes/pi05/launch_pi05_${MODE}.sh"
fi
