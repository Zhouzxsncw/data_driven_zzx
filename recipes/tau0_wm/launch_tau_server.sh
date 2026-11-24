#!/usr/bin/env bash
# τ0-WM TongBot 00036 EE 20D 本机服务端（端口 1244）
#
#   bash recipes/tau0_wm/launch_tau_server.sh
#   bash recipes/tau0_wm/launch_tau_server.sh --load-only
#   bash recipes/tau0_wm/launch_tau_server.sh --pretrained-only
#
# 00036 EE SFT 只要 step_N 的 diffusion_pytorch_model.bin（约 11G）。
# 不要拉 latest_accelerator_state。没有权重先:
#   bash recipes/tau0_wm/pull_tau_ckpt.sh
set -euo pipefail

STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STAGING"

if [[ -f "${STAGING}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${STAGING}/env.sh" || true
fi

PYTHON="${TAU_PYTHON:-${HOME}/miniconda3/envs/tau0wm/bin/python}"
CKPT="${TAU_CKPT:-/home/xk/VLA/h100_ckpts/tau0wm_tongbot_00036_ee}"
PORT="${TAU_PORT:-1244}"
DENOISE="${TAU_DENOISE:-10}"
N_EXEC="${TAU_N_EXEC:-8}"
IK_MODE="${TAU_IK_MODE:-abort_rest}"
IK_SOLVER="${TAU_IK_SOLVER:-realman}"
IK_TRAJ="${TAU_IK_TRAJ_OPT:-none}"
LOG="$STAGING/server_tau0wm.log"
PROMPT="${TAU_PROMPT:-With both hands, pick up the yellow banana and the yellow lemon and put them into the black box; then take the green vegetables and place them into the blue box.}"

EXTRA=()
for arg in "$@"; do
  EXTRA+=("$arg")
done

echo "=== τ0-WM TongBot EE 20D ==="
echo "Python: $PYTHON ($("$PYTHON" --version 2>&1))"
echo "Ckpt:   $CKPT"
echo "Port:   $PORT"
echo "denoise/n_exec: $DENOISE / $N_EXEC"
echo "IK:     infer 出 EE → 客户端读完 → ws solve_ik"
echo "Log:    $LOG"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null | head -1)"
echo "================================"

if [[ ! -x "$PYTHON" ]]; then
  echo "找不到 τ0-WM python: $PYTHON（先 bash recipes/tau0_wm/setup_env.sh）" >&2
  exit 1
fi

if [[ "${1:-}" == "--load-only" || "${1:-}" == "--pretrained-only" ]]; then
  exec "$PYTHON" -u "$STAGING/serve_tong_tau.py" \
    --ckpt "$CKPT" --port "$PORT" --denoise "$DENOISE" --n-exec "$N_EXEC" \
    --ik-mode "$IK_MODE" --ik-solver "$IK_SOLVER" --ik-traj-opt "$IK_TRAJ" \
    ${TAU_IK_CONFIG:+--ik-config "$TAU_IK_CONFIG"} \
    ${TAU_IK_MAX_STEP_DEG:+--ik-max-step-deg "$TAU_IK_MAX_STEP_DEG"} \
    --instruction "$PROMPT" \
    "${EXTRA[@]}"
fi

OLD=$(pgrep -f "$STAGING/serve_tong_tau.py" || true)
if [[ -n "${OLD}" ]]; then
  echo "杀掉旧 τ0-WM 服务: $OLD"
  echo "$OLD" | xargs kill -9 2>/dev/null || true
  sleep 2
fi

nohup "$PYTHON" -u "$STAGING/serve_tong_tau.py" \
  --ckpt "$CKPT" --port "$PORT" \
  --denoise "$DENOISE" --n-exec "$N_EXEC" \
  --ik-mode "$IK_MODE" --ik-solver "$IK_SOLVER" --ik-traj-opt "$IK_TRAJ" \
  ${TAU_IK_CONFIG:+--ik-config "$TAU_IK_CONFIG"} \
  ${TAU_IK_MAX_STEP_DEG:+--ik-max-step-deg "$TAU_IK_MAX_STEP_DEG"} \
  --instruction "$PROMPT" \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &
PID=$!
echo "Server PID: $PID"
echo "日志: $LOG"
echo "等模型加载出现 listening 后再开客户端:"
echo "  bash $STAGING/launch_tau_client.sh"
echo "  tail -f $LOG"
sleep 3
if ps -p "$PID" >/dev/null 2>&1; then
  echo "已启动 PID=$PID"
else
  echo "启动失败:"
  tail -40 "$LOG"
  exit 1
fi
