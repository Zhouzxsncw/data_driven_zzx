#!/usr/bin/env bash
# GigaWorld-Policy-0.5 TongBot 本机服务端（端口 1243）
#
#   bash recipes/gigaworld_policy/launch_gwp_server.sh
#   bash recipes/gigaworld_policy/launch_gwp_server.sh --load-only
#   bash recipes/gigaworld_policy/launch_gwp_server.sh --pretrained-only
#
# 00073 SFT 只要 transformer/（约 12G）。没有权重先:
#   bash recipes/gigaworld_policy/pull_gwp_ckpt.sh
set -euo pipefail

STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STAGING"

GWP_ROOT="${GWP_ROOT:-/home/xk/VLA/giga-world-policy}"
PYTHON="${GWP_PYTHON:-${HOME}/miniconda3/envs/gigaworld-policy/bin/python}"

if [[ -f "${STAGING}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${STAGING}/env.sh" || true
fi
if [[ -f "${GWP_ROOT}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${GWP_ROOT}/env.sh" || true
fi

CKPT="${GWP_CKPT:-/home/xk/VLA/h100_ckpts/gwp05_tongbot_00073}"
PORT="${GWP_PORT:-1243}"
DENOISE="${GWP_DENOISE:-10}"
LOG="$STAGING/server_gwp.log"
PROMPT="${GWP_PROMPT:-Using both hands, pick up the courier items from the green conveyor belt. Place the white courier bags into the left box and the brown courier boxes into the right box.}"

EXTRA=()
for arg in "$@"; do
  EXTRA+=("$arg")
done

echo "=== GigaWorld-Policy-0.5 TongBot 16D ==="
echo "Python: $PYTHON ($("$PYTHON" --version 2>&1))"
echo "Ckpt:   $CKPT"
echo "Port:   $PORT"
echo "denoise: $DENOISE"
echo "Log:    $LOG"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null | head -1)"
echo "================================"

if [[ ! -x "$PYTHON" ]]; then
  echo "找不到 GWP python: $PYTHON（先 bash recipes/gigaworld_policy/setup_env.sh）" >&2
  exit 1
fi

if [[ "${1:-}" == "--load-only" || "${1:-}" == "--pretrained-only" ]]; then
  exec "$PYTHON" -u "$STAGING/serve_tong_gwp.py" \
    --ckpt "$CKPT" --port "$PORT" --denoise "$DENOISE" \
    --instruction "$PROMPT" \
    "${EXTRA[@]}"
fi

OLD=$(pgrep -f "$STAGING/serve_tong_gwp.py" || true)
if [[ -n "${OLD}" ]]; then
  echo "杀掉旧 GWP 服务: $OLD"
  echo "$OLD" | xargs kill -9 2>/dev/null || true
  sleep 2
fi

nohup "$PYTHON" -u "$STAGING/serve_tong_gwp.py" \
  --ckpt "$CKPT" --port "$PORT" \
  --denoise "$DENOISE" \
  --instruction "$PROMPT" \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &
PID=$!
echo "Server PID: $PID"
echo "日志: $LOG"
echo "等模型加载出现 listening 后再开客户端:"
echo "  bash $STAGING/launch_gwp_client.sh"
echo "  tail -f $LOG"
sleep 3
if ps -p "$PID" >/dev/null 2>&1; then
  echo "已启动 PID=$PID"
  bash "$STAGING/../common/start_lower_machine.sh" || \
    echo "[WARN] 下位机未拉起。重试: bash $STAGING/../common/start_lower_machine.sh ；跳过: SKIP_LOWER_MACHINE=1" >&2
else
  echo "启动失败:"
  tail -40 "$LOG"
  exit 1
fi
