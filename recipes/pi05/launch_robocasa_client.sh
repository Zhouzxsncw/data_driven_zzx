#!/usr/bin/env bash
# RoboCasa 客户端。仿真 Python 用 $SIM_ROOT/envs/robocasa，不占用 π₀.₅ 的 JAX 环境。
#
#   source "$SIM_ROOT/env.sh" robocasa
#   PI05_EMBODIMENT=robocasa365 bash recipes/pi05/launch_pi05_server.sh
#   bash recipes/pi05/launch_robocasa_client.sh
set -euo pipefail
STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$STAGING/../.." && pwd)"
INFER_ROOT="${INFER_ROOT:-${REPO_ROOT}/infer}"
PYTHON="${ROBOCASA_PYTHON:-${SIM_ROOT:-}/envs/robocasa/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "[ERROR] 没有 RoboCasa Python：$PYTHON 。先设 SIM_ROOT 并跑 scripts/bootstrap_sim_infer.sh" >&2
  exit 1
fi
export PYTHONPATH="${INFER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
if [[ "${MUJOCO_GL:-}" != "osmesa" ]]; then
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
fi
exec "$PYTHON" -u "$STAGING/eval_robocasa_client.py" "$@"
