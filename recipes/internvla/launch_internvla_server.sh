#!/usr/bin/env bash
# InternVLA-A1.5 本机服务端（端口 1237）
#
#   bash recipes/internvla/launch_internvla_server.sh
#   bash recipes/internvla/launch_internvla_server.sh --load-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/env_infer.sh"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

CKPT="${INTERNVLA_CKPT:-/data/internvla/outputs/00036_finetune_bs32/checkpoints/002000/pretrained_model}"
VLM="${VLM_MODEL_PATH:-/data/internvla/ckpts/Qwen3.5-2B}"
PORT="${INTERNVLA_PORT:-1237}"
LOG="$ROOT/server_internvla.log"
PROMPT="${INTERNVLA_PROMPT:-With both hands, pick up the yellow banana and the yellow lemon and put them into the black box; then take the green vegetables and place them into the blue box.}"

echo "=== InternVLA-A1.5 ==="
echo "Python: $(which python) ($(python --version 2>&1))"
echo "Ckpt:   $CKPT"
echo "VLM:    $VLM"
echo "Port:   $PORT"
echo "Log:    $LOG"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null)"
echo "================================"

if [[ ! -d "$CKPT" ]]; then
  echo "权重不存在: $CKPT" >&2
  exit 1
fi

if [[ "${1:-}" == "--load-only" ]]; then
  exec python -u "$ROOT/serve_tong_internvla.py" --ckpt "$CKPT" --vlm "$VLM" --load-only
fi

OLD=$(pgrep -f "$ROOT/serve_tong_internvla.py" || true)
if [[ -n "${OLD}" ]]; then
  echo "杀掉旧 InternVLA 服务: $OLD"
  echo "$OLD" | xargs kill -9 2>/dev/null || true
  sleep 2
fi

nohup python -u "$ROOT/serve_tong_internvla.py" \
  --ckpt "$CKPT" --vlm "$VLM" --port "$PORT" --instruction "$PROMPT" \
  > "$LOG" 2>&1 &
PID=$!
echo "Server PID: $PID"
echo "日志: $LOG"
echo "等模型加载出现 listening 后再开客户端:"
echo "  bash $ROOT/launch_internvla_client.sh"
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
