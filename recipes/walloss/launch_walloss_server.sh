#!/usr/bin/env bash
# WALL-OSS 本机服务端（端口 1239）
#
#   bash recipes/walloss/launch_walloss_server.sh
#   bash recipes/walloss/launch_walloss_server.sh --load-only
#   bash recipes/walloss/launch_walloss_server.sh --openloop
set -euo pipefail

STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STAGING"
# env.sh 会 export ROOT=/data/wall-oss，这里不能用 ROOT 当脚本目录。
# shellcheck disable=SC1091
source "$STAGING/env_infer.sh"

CKPT="${WALLOSS_CKPT:-/data/wall-oss/outputs/walloss05_tongbot_ee_00086/9}"
PORT="${WALLOSS_PORT:-1239}"
N_EXEC="${WALLOSS_N_EXEC:-8}"
IK_MODE="${WALLOSS_IK_MODE:-abort_rest}"
IK_SOLVER="${WALLOSS_IK_SOLVER:-realman}"
IK_TRAJ="${WALLOSS_IK_TRAJ_OPT:-none}"
LOG="$STAGING/server_walloss.log"
PROMPT="${WALLOSS_PROMPT:-Using both hands, pull open the drawer with the blue handle using your left hand, take out the white medicine box with your right hand, place it into the green drawer, and finally close the drawer.}"

echo "=== WALL-OSS ==="
echo "Python: $(which python) ($(python --version 2>&1))"
echo "Ckpt:   $CKPT"
echo "Port:   $PORT"
echo "n_exec: $N_EXEC"
echo "ik:     $IK_MODE / $IK_SOLVER / traj=$IK_TRAJ"
echo "Log:    $LOG"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null)"
echo "================================"

if [[ ! -f "${CKPT}/model.safetensors" ]]; then
  echo "权重不存在: ${CKPT}/model.safetensors" >&2
  exit 1
fi

if [[ "${1:-}" == "--load-only" ]]; then
  exec python -u "$STAGING/serve_tong_walloss.py" --ckpt "$CKPT" --n-exec "$N_EXEC" --load-only
fi

if [[ "${1:-}" == "--openloop" ]]; then
  exec python -u "$STAGING/openloop_00086.py" --ckpt "$CKPT" --n-frames "${2:-4}"
fi

OLD=$(pgrep -f "$STAGING/serve_tong_walloss.py" || true)
if [[ -n "${OLD}" ]]; then
  echo "杀掉旧 WALL-OSS 服务: $OLD"
  echo "$OLD" | xargs kill -9 2>/dev/null || true
  sleep 2
fi

nohup python -u "$STAGING/serve_tong_walloss.py" \
  --ckpt "$CKPT" --port "$PORT" --instruction "$PROMPT" --n-exec "$N_EXEC" \
  --ik-mode "$IK_MODE" \
  --ik-solver "$IK_SOLVER" \
  --ik-traj-opt "$IK_TRAJ" \
  ${WALLOSS_IK_CONFIG:+--ik-config "$WALLOSS_IK_CONFIG"} \
  ${WALLOSS_IK_MAX_STEP_DEG:+--ik-max-step-deg "$WALLOSS_IK_MAX_STEP_DEG"} \
  > "$LOG" 2>&1 &
PID=$!
echo "Server PID: $PID"
echo "日志: $LOG"
echo "等模型加载出现 listening 后再开客户端:"
echo "  bash $STAGING/launch_walloss_client.sh"
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
