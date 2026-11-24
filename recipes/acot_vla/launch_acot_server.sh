#!/usr/bin/env bash
# ACoT-VLA 推理服务端（gpu-master 端口 8000）
# 先跑这个，再跑 launch_acot_client.sh
set -euo pipefail

ACOT_ROOT="/home/xk/ACoT-VLA"
CKPT="checkpoints/acot_tongbot_00036/tongbot_00036_run1/4000"
PORT="${ACOT_PORT:-8000}"

if ss -tln | grep -q ":${PORT}"; then
  echo "ACoT-VLA 服务已在端口 ${PORT} 运行"
  exit 0
fi

cd "$ACOT_ROOT"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0"

echo "=== ACoT-VLA 服务端 ==="
echo "Root:     $ACOT_ROOT"
echo "Checkpoint: $CKPT"
echo "Port:     $PORT"
echo "Log:      $ACOT_ROOT/serve_acot.log"
echo "======================="

nohup uv run --no-sync python scripts/serve_policy.py \
  --port "$PORT" \
  policy:checkpoint \
  --policy.config acot_tongbot_00036 \
  --policy.dir "$CKPT" \
  > serve_acot.log 2>&1 &

echo "PID: $! — 等待就绪..."
for i in $(seq 1 30); do
  sleep 2
  if ss -tln | grep -q ":${PORT}"; then
    echo "✅ 服务已就绪: ws://0.0.0.0:${PORT}"
    bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../common" && pwd)/start_lower_machine.sh" || \
      echo "[WARN] 下位机未拉起。SKIP_LOWER_MACHINE=1 可跳过。" >&2
    exit 0
  fi
done
echo "❌ 30s 内未就绪，查日志: tail -f $ACOT_ROOT/serve_acot.log"
exit 1
