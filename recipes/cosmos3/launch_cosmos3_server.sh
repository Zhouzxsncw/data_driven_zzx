#!/usr/bin/env bash
# Cosmos3 TongBot 00073 SFT 本机服务端（端口 1236，避开 VA 1234 / DreamZero 1235）
#
#   bash recipes/cosmos3/launch_cosmos3_server.sh
#   bash recipes/cosmos3/launch_cosmos3_server.sh --pretrained-only
#   bash recipes/cosmos3/launch_cosmos3_server.sh --load-only   # 只测预训练加载
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/env_infer.sh"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${COSMOS3_MODEL:-/home/xk/VLA/cosmos_h100_setup/Cosmos3-Nano}"
CKPT="${COSMOS3_CKPT:-/home/xk/VLA/h100_ckpts/cosmos3_00073/iter_000001500}"
VAE="${WAN_VAE_PATH:-/home/xk/VLA/dreamzero_staging/pretrained/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
PORT="${COSMOS3_PORT:-1236}"
STEPS="${COSMOS3_STEPS:-10}"
LOG="$ROOT/server_cosmos3.log"
PROMPT="${COSMOS3_PROMPT:-Using both hands, pick up the courier items from the green conveyor belt. Place the white courier bags into the left box and the brown courier boxes into the right box.}"
PRETRAINED_ONLY=0
if [[ "${1:-}" == "--pretrained-only" ]]; then
  PRETRAINED_ONLY=1
  shift
fi

export PYTHONPATH="/home/xk/VLA/cosmos-framework:${PYTHONPATH:-}"
export WAN_VAE_PATH="$VAE"
export COSMOS3_CKPT="$CKPT"

echo "=== Cosmos3 TongBot SFT ==="
echo "Python: $(which python) ($(python --version 2>&1))"
echo "SFT:    $CKPT"
echo "VAE:    $VAE"
echo "Nano:   $MODEL"
echo "Port:   $PORT"
echo "Steps:  $STEPS  (默认 10 + 跳过 VAE decode；每包落 vision latent 供离线 decode)"
echo "Latent: ${COSMOS3_LATENT_DIR:-$CKPT/vision_latents}"
echo "Log:    $LOG"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null)"
echo "================================"

if [[ "$PRETRAINED_ONLY" -eq 1 ]]; then
  if [[ ! -d "$MODEL/transformer" ]]; then
    echo "预训练权重不存在: $MODEL" >&2
    exit 1
  fi
elif [[ ! -d "$CKPT/model" ]]; then
  echo "SFT DCP 不存在: $CKPT/model" >&2
  exit 1
fi

if [[ "${1:-}" == "--load-only" ]]; then
  exec python -u "$ROOT/smoke_t2i.py" --model "$MODEL" --load-only
fi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ "${FREE_MIB:-0}" -lt 35000 ]]; then
  echo "空闲显存约 ${FREE_MIB} MiB，Nano bf16 大约还要 32GB+。建议先停占用卡的进程。" >&2
fi

OLD=$(pgrep -f "$ROOT/serve_tong_cosmos3.py" || true)
if [[ -n "${OLD}" ]]; then
  echo "杀掉旧 Cosmos3 服务: $OLD"
  echo "$OLD" | xargs kill -9 2>/dev/null || true
  sleep 2
fi

SERVE_ARGS=(--model "$MODEL" --ckpt "$CKPT" --vae "$VAE" --port "$PORT" --instruction "$PROMPT" --steps "$STEPS")
if [[ "$PRETRAINED_ONLY" -eq 1 ]]; then
  SERVE_ARGS+=(--pretrained-only)
fi
# 其余参数原样转给 serve，例如 --decode-video / --guidance-scale 1
if [[ "$#" -gt 0 ]]; then
  SERVE_ARGS+=("$@")
fi

PYTHON_BIN="${COSMOS3_PYTHON:-/data/cosmos3-infer/bin/python}"
nohup "$PYTHON_BIN" -u "$ROOT/serve_tong_cosmos3.py" \
  "${SERVE_ARGS[@]}" \
  > "$LOG" 2>&1 &
PID=$!
echo "Server PID: $PID"
echo "日志: $LOG"
echo "等 SFT DCP 加载（约 3–8 分钟），出现 listening 后再开客户端:"
echo "  bash $ROOT/launch_cosmos3_client.sh"
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
