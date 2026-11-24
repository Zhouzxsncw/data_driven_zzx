#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/tk/data_Driven
export PI05_SERVER_LOG="$ROOT/recipes/pi05/server_117_120_121_125_step013740_tk.log"
export SKIP_LOWER_MACHINE="${SKIP_LOWER_MACHINE:-1}"
exec bash "$ROOT/recipes/pi05/run_tk_pi05.sh" "${1:-check}" \
  "$ROOT/infer/infer/config/lerobot/pi05_117_120_121_125_step013740_tk_ws.yaml" \
  "/media/tk/T9_7/checkpoint/rm03_117_120_121_125_PI05_cty19_ah10_cty19_step013740/pretrained_model"
