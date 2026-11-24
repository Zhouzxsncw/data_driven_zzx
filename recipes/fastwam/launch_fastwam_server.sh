#!/usr/bin/env bash
# FastWAM 本机服务端（端口 1240）
#
#   bash recipes/fastwam/launch_fastwam_server.sh
#   bash recipes/fastwam/launch_fastwam_server.sh --load-only
set -euo pipefail

STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STAGING"

# FastWAM 环境
FASTWAM_ROOT="${FASTWAM_ROOT:-/home/xk/VLA/FastWAM}"
PYTHON="${FASTWAM_PYTHON:-${HOME}/miniconda3/envs/fastwam/bin/python}"

# 加载 env.sh 获取环境变量（HF/DIFFSYNTH 等），但不依赖 conda activate 切 PATH
if [[ -f "${FASTWAM_ROOT}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${FASTWAM_ROOT}/env.sh" || true
fi

CKPT="${FASTWAM_CKPT:-${FASTWAM_ROOT}/runs/tongbot_00073_3cam384_1e-4/2026-09-08_19-57-50/checkpoints/weights/step_025000.pt}"
DATASET_STATS="${FASTWAM_DATASET_STATS:-${FASTWAM_ROOT}/runs/tongbot_00073_3cam384_1e-4/2026-09-08_19-57-50/dataset_stats.json}"
PORT="${FASTWAM_PORT:-1240}"
ACTION_HORIZON="${FASTWAM_ACTION_HORIZON:-32}"
DENOISE="${FASTWAM_DENOISE:-15}"
COMPILE="${FASTWAM_COMPILE:-}"  # 设置非空字符串启用 torch.compile
LOG="$STAGING/server_fastwam_00073.log"
PROMPT="${FASTWAM_PROMPT:-Using both hands, pick up the courier items from the green conveyor belt. Place the white courier bags into the left box and the brown courier boxes into the right box.}"

export PYTHONPATH="${FASTWAM_ROOT}:${FASTWAM_ROOT}/src:${PYTHONPATH:-}"

echo "=== FastWAM 00073 ==="
echo "Python: $PYTHON ($(${PYTHON} --version 2>&1))"
echo "Ckpt:   $CKPT"
echo "Stats:  $DATASET_STATS"
echo "Port:   $PORT"
echo "action_horizon: $ACTION_HORIZON"
echo "denoise: $DENOISE"
echo "compile: ${COMPILE:-off}"
echo "Log:    $LOG"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null)"
echo "================================"

if [[ ! -f "${CKPT}" ]]; then
  echo "权重不存在: ${CKPT}" >&2
  exit 1
fi

# 构建参数
SERVE_ARGS=(
  --ckpt "$CKPT" --port "$PORT"
  --action-horizon "$ACTION_HORIZON" --denoise "$DENOISE"
  --instruction "$PROMPT"
  --dataset-stats "$DATASET_STATS"
)
if [[ -n "${COMPILE}" ]]; then
  SERVE_ARGS+=(--compile)
fi

if [[ "${1:-}" == "--load-only" ]]; then
  exec "$PYTHON" -u "$STAGING/serve_tong_fastwam.py" "${SERVE_ARGS[@]}" --load-only
fi

OLD=$(pgrep -f "$STAGING/serve_tong_fastwam.py" || true)
if [[ -n "${OLD}" ]]; then
  echo "杀掉旧 FastWAM 服务: $OLD"
  echo "$OLD" | xargs kill -9 2>/dev/null || true
  sleep 2
fi

nohup "$PYTHON" -u "$STAGING/serve_tong_fastwam.py" "${SERVE_ARGS[@]}" \
  > "$LOG" 2>&1 &
PID=$!
echo "Server PID: $PID"
echo "日志: $LOG"
echo "等模型加载出现 listening 后再开客户端:"
echo "  bash $STAGING/launch_fastwam_client.sh"
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
