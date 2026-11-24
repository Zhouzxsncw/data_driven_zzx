#!/usr/bin/env bash
# Motus 本机服务端（端口 1241）
#
#   bash recipes/motus/launch_motus_server.sh
#   bash recipes/motus/launch_motus_server.sh --load-only
#
#   # 开启 dream video 落盘（每包 VAE decode，会增加延迟）：
#   MOTUS_DUMP_VIDEO=/home/xk/VLA/h100_ckpts/motus_dream bash recipes/motus/launch_motus_server.sh
set -euo pipefail

STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STAGING"

# Motus 环境（conda motus, torch 2.9.0+cu128）
MOTUS_ROOT="${MOTUS_ROOT:-/home/xk/VLA/Motus}"
PYTHON="${MOTUS_PYTHON:-${HOME}/miniconda3/envs/motus/bin/python}"

if [[ -f "${MOTUS_ROOT}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${MOTUS_ROOT}/env.sh" || true
fi

CKPT="${MOTUS_CKPT:-/home/xk/VLA/h100_ckpts/motus_tongbot_jiangsuan4/checkpoint_step_9000}"
PORT="${MOTUS_PORT:-1241}"
DENOISE="${MOTUS_DENOISE:-10}"
LOG="$STAGING/server_motus.log"
PROMPT="${MOTUS_PROMPT:-Use both arms to complete the pick and place task.}"
DUMP_VIDEO="${MOTUS_DUMP_VIDEO:-}"  # 非空路径 = 开启 dream video 落盘
CAM_MODE="${MOTUS_CAM_MODE:-front}"  # front=js4 旧 ckpt；three=之后三路 T 拼图 ckpt

echo "=== Motus TongBot 16D ==="
echo "Python: $PYTHON ($(${PYTHON} --version 2>&1))"
echo "Ckpt:   $CKPT"
echo "Port:   $PORT"
echo "denoise: $DENOISE"
echo "Log:    $LOG"
echo "Dump:   ${DUMP_VIDEO:-off}"
echo "Cams:   ${CAM_MODE}"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null)"
echo "================================"

if [[ ! -f "${CKPT}/mp_rank_00_model_states.pt" ]]; then
  echo "权重不存在: ${CKPT}/mp_rank_00_model_states.pt" >&2
  exit 1
fi

if [[ "${1:-}" == "--load-only" ]]; then
  exec "$PYTHON" -u "$STAGING/serve_tong_motus.py" \
    --ckpt "$CKPT" --denoise "$DENOISE" --cam-mode "$CAM_MODE" --load-only
fi

OLD=$(pgrep -f "$STAGING/serve_tong_motus.py" || true)
if [[ -n "${OLD}" ]]; then
  echo "杀掉旧 Motus 服务: $OLD"
  echo "$OLD" | xargs kill -9 2>/dev/null || true
  sleep 2
fi

DUMP_ARGS=""
if [[ -n "${DUMP_VIDEO}" ]]; then
  mkdir -p "${DUMP_VIDEO}" 2>/dev/null || true
  DUMP_ARGS="--dump-video ${DUMP_VIDEO}"
fi

nohup "$PYTHON" -u "$STAGING/serve_tong_motus.py" \
  --ckpt "$CKPT" --port "$PORT" \
  --denoise "$DENOISE" \
  --instruction "$PROMPT" \
  --cam-mode "$CAM_MODE" \
  ${DUMP_ARGS} \
  > "$LOG" 2>&1 &
PID=$!
echo "Server PID: $PID"
echo "日志: $LOG"
echo "等模型加载出现 listening 后再开客户端:"
echo "  bash $STAGING/launch_motus_client.sh"
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
