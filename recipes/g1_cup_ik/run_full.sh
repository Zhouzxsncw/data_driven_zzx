#!/usr/bin/env bash
# 本机一整套：采集（杯子+臂初值随机）→ 微调 GR00T N1.7 SONIC → 并行评测（显存自动堆 env）。
# 不杀已有 π0.5 / DreamZero / τ0-WM 推理服务；显存不够就等。
export TERM=xterm-256color
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
# shellcheck source=../gr00t_sonic/env.sh
source "${SCRIPT_DIR}/../gr00t_sonic/env.sh"

DATA_OUT="${DATA_OUT:-${HOME}/VLA/datasets/g1_cup_ik_sonic_full}"
TRAIN_ROOT="${TRAIN_ROOT:-${HOME}/VLA/gr00t_sonic_outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_cup_ik_sonic}"
EXP_DIR="${TRAIN_ROOT}/${EXPERIMENT_NAME}"
NUM_SUCCESS="${NUM_SUCCESS:-200}"
NUM_ENVS_COLLECT="${NUM_ENVS_COLLECT:-32}"
MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
COLLECT_FREE_MIB="${COLLECT_FREE_MIB:-40000}"
TRAIN_FREE_MIB="${TRAIN_FREE_MIB:-50000}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
mkdir -p "${EXP_DIR}" "${DATA_OUT}"
LOG="${EXP_DIR}/pipeline.log"

gpu_free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '
}

wait_free() {
    local need="$1"
    local label="$2"
    echo "[pipeline] wait ${label} free>=${need} MiB（不杀其它进程）" | tee -a "${LOG}"
    local i
    for i in $(seq 1 1440); do
        local free
        free="$(gpu_free_mib)"
        echo "[pipeline] ${label} free=${free} MiB  $(date '+%F %T')" | tee -a "${LOG}"
        if [ "${free}" -ge "${need}" ]; then
            return 0
        fi
        sleep 30
    done
    echo "[pipeline] ${label} 等显存超时" | tee -a "${LOG}"
    return 1
}

echo "[pipeline] start $(date '+%F %T') log=${LOG}" | tee -a "${LOG}"
echo "[pipeline] data=${DATA_OUT} exp=${EXP_DIR}" | tee -a "${LOG}"

if [ "${SKIP_COLLECT}" != "1" ]; then
    wait_free "${COLLECT_FREE_MIB}" collect
    echo "[pipeline] collect ${NUM_SUCCESS} eps arm_init+cup_rand" | tee -a "${LOG}"
    NUM_ENVS="${NUM_ENVS_COLLECT}" NUM_SUCCESS="${NUM_SUCCESS}" OUT="${DATA_OUT}" \
        bash "${SCRIPT_DIR}/collect.sh" >> "${LOG}" 2>&1
    echo "[pipeline] collect done $(date '+%F %T')" | tee -a "${LOG}"
else
    echo "[pipeline] skip collect, use ${DATA_OUT}" | tee -a "${LOG}"
fi

if [ ! -f "${DATA_OUT}/meta/modality.json" ]; then
    echo "[pipeline] 采集产物不像 LeRobot：${DATA_OUT}" | tee -a "${LOG}"
    exit 1
fi

if [ "${SKIP_TRAIN}" != "1" ]; then
    wait_free "${TRAIN_FREE_MIB}" train
    echo "[pipeline] train steps=${MAX_STEPS} bs=${GLOBAL_BATCH_SIZE}" | tee -a "${LOG}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export GR00T_USE_RELATIVE_ACTION=0
    export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${HOME}/VLA/Isaac-GR00T/checkpoints/Cosmos-Reason2-2B}"
    TRAIN_SH="${SCRIPT_DIR}/../gr00t_sonic/train.sh"
    if [ "${UNITREE_4090:-0}" = "1" ]; then
        TRAIN_SH="${SCRIPT_DIR}/../gr00t_sonic/train_4090.sh"
    fi
    bash "${TRAIN_SH}" \
        --dataset-path "${DATA_OUT}" \
        --output-dir "${TRAIN_ROOT}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        --max-steps "${MAX_STEPS}" \
        --save-steps "${SAVE_STEPS}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        >> "${LOG}" 2>&1
    echo "[pipeline] train done $(date '+%F %T')" | tee -a "${LOG}"
else
    echo "[pipeline] skip train" | tee -a "${LOG}"
fi

if [ "${SKIP_EVAL}" != "1" ]; then
    echo "[pipeline] eval auto-envs" | tee -a "${LOG}"
    EXP_DIR="${EXP_DIR}" \
        bash "${SCRIPT_DIR}/eval.sh" >> "${LOG}" 2>&1
    echo "[pipeline] eval done $(date '+%F %T')" | tee -a "${LOG}"
    if [ -f "${EXP_DIR}/eval_"*/eval_summary.json ]; then
        echo "[pipeline] summary:" | tee -a "${LOG}"
        ls -d "${EXP_DIR}"/eval_checkpoint-* 2>/dev/null | tee -a "${LOG}" || true
    fi
fi

echo "[pipeline] ALL DONE $(date '+%F %T')" | tee -a "${LOG}"
