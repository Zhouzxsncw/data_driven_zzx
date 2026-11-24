#!/usr/bin/env bash
# MemoryVLA 本机服务端（端口 1238）
#
#   bash recipes/memoryvla/launch_memoryvla_server.sh
#   bash recipes/memoryvla/launch_memoryvla_server.sh --load-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/env_infer.sh"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

CKPT="${MEMVLA_CKPT:-/data/memoryvla/outputs/00086/memvla_00086_80k--image_aug/checkpoints/step-010000-epoch-00-loss=0.0135.pt}"
PORT="${MEMORYVLA_PORT:-1238}"
LOG="$ROOT/server_memoryvla.log"
PROMPT="${MEMORYVLA_PROMPT:-Using both hands, pull open the drawer with the blue handle using your left hand, take out the white medicine box with your right hand, place it into the green drawer, and finally close the drawer.}"

echo "=== MemoryVLA ==="
echo "Python: $(which python) ($(python --version 2>&1))"
echo "Ckpt:   $CKPT"
echo "Port:   $PORT"
echo "Log:    $LOG"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null)"
echo "================================"

if [[ ! -f "$CKPT" ]]; then
  echo "权重不存在: $CKPT" >&2
  exit 1
fi

if [[ "${1:-}" == "--load-only" ]]; then
  exec python -u "$ROOT/serve_tong_memoryvla.py" --ckpt "$CKPT" --bf16 --load-only
fi

OLD=$(pgrep -f "$ROOT/serve_tong_memoryvla.py" || true)
if [[ -n "${OLD}" ]]; then
  echo "杀掉旧 MemoryVLA 服务: $OLD"
  echo "$OLD" | xargs kill -9 2>/dev/null || true
  sleep 2
fi

nohup python -u "$ROOT/serve_tong_memoryvla.py" \
  --ckpt "$CKPT" --port "$PORT" --instruction "$PROMPT" --bf16 \
  > "$LOG" 2>&1 &
PID=$!
echo "Server PID: $PID"
echo "日志: $LOG"
echo "等模型加载出现 listening 后再开客户端:"
echo "  bash $ROOT/launch_memoryvla_client.sh"
echo "  tail -f $LOG"
sleep 3
if ps -p "$PID" >/dev/null 2>&1; then
  echo "已启动 PID=$PID"
  bash "$ROOT/../common/start_lower_machine.sh" || \
    echo "[WARN] 下位机未拉起。重试: bash $ROOT/../common/start_lower_machine.sh ；跳过: SKIP_LOWER_MACHINE=1" >&2
else
  echo "启动失败:"
  tail -40 "$LOG"
  exit 1
fi
