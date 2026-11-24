#!/usr/bin/env bash
# 本机不接真机：同一份 YAML 起服务（可选）+ websocket 冒烟客户端。
#
#   bash recipes/common/launch_e2e.sh pi05_00086_js16_22445 --pair-only
#   SKIP_LOWER_MACHINE=1 bash recipes/common/launch_e2e.sh pi05_00086_js16_22445 --start-server
#   bash recipes/common/launch_e2e.sh pi05_00086_js16_22445 --validate --n-samples 24
#
# 真机仍走 recipes/common/launch_client.sh（同一 YAML）。开环 VALIDATION 不接真机。
set -euo pipefail
unalias -a 2>/dev/null || true

COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$COMMON/../.." && pwd)"
INFER_ROOT="${INFER_ROOT:-$REPO_ROOT/infer}"

_alias_yaml() {
  case "$1" in
    pi05_00086_js16_22445) echo "infer/infer/config/lerobot/pi05_00086_4gpu_bs384_ah100_step22445_ws.yaml" ;;
    pi05_00036_js8_ah15) echo "infer/infer/config/lerobot/pi05_00036_js8_bs384_ah15_fixbase_step7860_ws.yaml" ;;
    pi05_00036_js8_ah100) echo "infer/infer/config/lerobot/pi05_00036_js8_bs384_ah100_fixbase_step7860_ws.yaml" ;;
    walloss_00036_js16) echo "infer/infer/config/lingbot/walloss_tongbot_ee_00036_js16_ep9_local.yaml" ;;
    *) echo "" ;;
  esac
}

START_SERVER=0
PAIR_ONLY=0
VALIDATE=0
N_INFER=2
N_SAMPLES=24
DATASET=""
OUT=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-server) START_SERVER=1; shift ;;
    --pair-only) PAIR_ONLY=1; shift ;;
    --validate) VALIDATE=1; shift ;;
    --n) N_INFER="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help)
      echo "用法: bash recipes/common/launch_e2e.sh <yaml|短名> [--pair-only] [--start-server] [--n 2]"
      echo "       [--validate] [--n-samples 24] [--dataset PATH] [--out json]"
      echo "短名: pi05_00086_js16_22445  pi05_00036_js8_ah15  pi05_00036_js8_ah100  walloss_00036_js16"
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ ${#ARGS[@]} -lt 1 ]]; then
  echo "缺少 YAML/短名。 bash recipes/common/launch_e2e.sh --help" >&2
  exit 2
fi
CONFIG_ARG="${ARGS[0]}"

alias_rel="$(_alias_yaml "$CONFIG_ARG")"
if [[ -n "$alias_rel" ]]; then
  CONFIG_ARG="$alias_rel"
fi
CONFIG=""
for c in "$CONFIG_ARG" "$REPO_ROOT/$CONFIG_ARG" "$REPO_ROOT/infer/$CONFIG_ARG"; do
  if [[ -f "$c" ]]; then
    CONFIG="$(python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$c")"
    break
  fi
done
if [[ -z "$CONFIG" ]]; then
  echo "[ERROR] 找不到 YAML: ${ARGS[0]}" >&2
  exit 2
fi

PY="${OPENPI_INFER_PYTHON:-${LINGBOTVA_PYTHON:-/home/xk/miniconda3/envs/lingbotva/bin/python}}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

export PYTHONPATH="${INFER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
echo "=== e2e pair-check ==="
echo "YAML: $CONFIG"
"$PY" "$COMMON/e2e_ws_client.py" --config "$CONFIG" --pair-only
if [[ "$PAIR_ONLY" == 1 ]]; then
  exit 0
fi

PTYPE="$("$PY" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
raw = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
pol = raw.get("policy") or {}
print(str(pol.get("type") or "").strip().lower())
print(str(pol.get("ckpt_dir") or "").strip())
print(str(pol.get("lingbot_port") or "").strip())
print(str(pol.get("task_prompt") or "").strip())
PY
)"
TYPE="$(printf '%s\n' "$PTYPE" | sed -n '1p')"
CKPT="$(printf '%s\n' "$PTYPE" | sed -n '2p')"
PORT="$(printf '%s\n' "$PTYPE" | sed -n '3p')"
PROMPT="$(printf '%s\n' "$PTYPE" | sed -n '4p')"

if [[ "$START_SERVER" == 1 ]]; then
  export SKIP_LOWER_MACHINE=1
  echo "=== start server (SKIP_LOWER_MACHINE=1) ==="
  if [[ "$TYPE" == "pi05" ]]; then
    PI05_CONFIG="$CONFIG" PI05_PORT="${PI05_PORT:-$PORT}" \
      bash "$REPO_ROOT/recipes/pi05/launch_pi05_server.sh"
  elif [[ "$TYPE" == "walloss" ]]; then
    WALLOSS_CKPT="$CKPT" WALLOSS_PORT="${WALLOSS_PORT:-$PORT}" \
      WALLOSS_PROMPT="${WALLOSS_PROMPT:-$PROMPT}" \
      bash "$REPO_ROOT/recipes/walloss/launch_walloss_server.sh"
  else
    echo "[ERROR] 不会起 type=$TYPE 的服务" >&2
    exit 2
  fi
  echo "等 listening …"
  for i in $(seq 1 90); do
    if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
      echo "port $PORT listening"
      break
    fi
    sleep 2
    if [[ "$i" == 90 ]]; then
      echo "[ERROR] ${PORT} 一直没 listening" >&2
      exit 1
    fi
  done
fi

if [[ "$VALIDATE" == 1 ]]; then
  echo "=== dataset openloop validate × ${N_SAMPLES} ==="
  extra=()
  [[ -n "$DATASET" ]] && extra+=(--dataset "$DATASET")
  [[ -n "$OUT" ]] && extra+=(--out "$OUT")
  exec "$PY" "$COMMON/dataset_validate.py" --config "$CONFIG" --n-samples "$N_SAMPLES" "${extra[@]}"
fi

echo "=== e2e infer × ${N_INFER} ==="
exec "$PY" "$COMMON/e2e_ws_client.py" --config "$CONFIG" --n "$N_INFER"
