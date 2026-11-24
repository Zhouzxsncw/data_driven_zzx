#!/usr/bin/env bash
# GPU 并行差分 IK 抓杯 → GR00T SONIC LeRobot v2.1（ego_view + 46D state + 78D action）。
# 带相机默认 32 env；无相机旧 npz 那套 256 会爆显存。
# 烟雾：NUM_ENVS=8 NUM_SUCCESS=2 bash collect.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
activate_isaaclab

NUM_ENVS="${NUM_ENVS:-${NUM_ENVS_COLLECT:-32}}"
NUM_SUCCESS="${NUM_SUCCESS:-2000}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${OUT:-${G1_CUP_SONIC_OUT}}"
HEADLESS="${HEADLESS:-1}"
# Blackwell + Isaac 5.1：--enable_cameras 会 hydra crash，默认光栅 ego_view
ENABLE_CAMERAS="${ENABLE_CAMERAS:-0}"
mkdir -p "${OUT}"

EXTRA_ARGS=()
if [ "${HEADLESS}" = "1" ]; then
    EXTRA_ARGS+=(--headless)
fi
if [ "${ENABLE_CAMERAS}" = "1" ]; then
    EXTRA_ARGS+=(--enable_cameras)
fi

cd "${ISAACLAB_DIR}"
exec ./isaaclab.sh -p "${SCRIPT_DIR}/collect_gpu_ik.py" \
    --device "${DEVICE}" \
    --num_envs "${NUM_ENVS}" \
    --num_success "${NUM_SUCCESS}" \
    --out_dir "${OUT}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
