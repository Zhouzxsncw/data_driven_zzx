#!/usr/bin/env bash
# 本机 LingBot-VA 00073 推理服务（江算 16 卡 w30 step500）
# 权重: /home/xk/VLA/h100_ckpts/00073_va_16gpu_w30/checkpoint_step_500
# GPU: NVIDIA RTX PRO 6000 96GB
#
#   bash recipes/lingbot_va/launch_00073_server.sh
#   bash recipes/lingbot_va/launch_00073_server.sh --test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${LINGBOTVA_PYTHON:-/home/xk/miniconda3/envs/lingbotva/bin/python}"
CONFIG="${VA_CONFIG:-$ROOT/config_00073.yaml}"
WEIGHT="/home/xk/VLA/h100_ckpts/00073_va_16gpu_w30/checkpoint_step_500/transformer/diffusion_pytorch_model.safetensors"
MODE="${1:-server}"
if [[ "$MODE" == "--test" ]]; then
  MODE=test
fi

# 服务端不要 PYTHONNOUSERSITE=1：会让 requests 的包元数据读不到
unset PYTHONNOUSERSITE || true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export LINGBOT_VA_CODE="${LINGBOT_VA_CODE:-/home/xk/VLA/lingbot-va}"
export LINGBOT_VA_RESOLVED_ROOT="${LINGBOT_VA_RESOLVED_ROOT:-/home/xk/VLA/lingbot-va-pretrained/_resolved_00073_va}"
export LINGBOT_VA_DEBUG_LOG="${LINGBOT_VA_DEBUG_LOG:-$ROOT/va_infer_debug.jsonl}"
export PYTHONUNBUFFERED=1

echo "=== LingBot-VA 00073 16gpu w30 step500 ==="
echo "Python: $PYTHON ($("$PYTHON" --version 2>&1))"
echo "Code:   $LINGBOT_VA_CODE"
echo "Config: $CONFIG"
echo "Weight: $WEIGHT"
echo "Mode:   $MODE"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null)"
echo "================================"

if [[ ! -x "$PYTHON" ]]; then
  echo "找不到 lingbotva python: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$LINGBOT_VA_CODE/wan_va/wan_va_server.py" ]]; then
  echo "找不到代码 $LINGBOT_VA_CODE" >&2
  exit 1
fi
if [[ ! -f "$WEIGHT" ]]; then
  echo "权重未就绪: $WEIGHT" >&2
  echo "确认权重存在: $WEIGHT" >&2
  exit 1
fi

if [[ "$MODE" == "server" ]]; then
  OLD_PIDS=$(pgrep -f "$ROOT/run_lingbotva_infer.py" || true)
  if [[ -n "${OLD_PIDS}" ]]; then
    echo "杀掉旧进程: $OLD_PIDS"
    echo "$OLD_PIDS" | xargs kill -9 2>/dev/null || true
    sleep 2
  fi
  nohup "$PYTHON" "$ROOT/run_lingbotva_infer.py" --config "$CONFIG" --mode server \
    > "$ROOT/server_00073.log" 2>&1 &
  PID=$!
  echo "Server PID: $PID"
  echo "日志: $ROOT/server_00073.log"
  sleep 8
  if ps -p "$PID" >/dev/null 2>&1; then
    echo "已启动 PID=$PID ，模型加载还要 1–2 分钟，看日志出现 listening / serve"
    ss -tlnp 2>/dev/null | grep 1234 || echo "端口 1234 尚未就绪（加载中）"
    bash "$ROOT/../common/start_lower_machine.sh" || \
      echo "[WARN] 下位机未拉起。重试: bash $ROOT/../common/start_lower_machine.sh ；跳过: SKIP_LOWER_MACHINE=1" >&2
  else
    echo "启动失败:"
    tail -40 "$ROOT/server_00073.log"
    exit 1
  fi
else
  exec "$PYTHON" "$ROOT/run_lingbotva_infer.py" --config "$CONFIG" --mode "$MODE"
fi
