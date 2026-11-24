#!/usr/bin/env bash
# 等 h100-103 上的 τ0-WM 00073 300k 收口（进程退、GPU 2-7 空），再开 00091 300k。
# 不杀 00073、不抢卡。00091 数据转换另跑 convert_00091_ee_on_h100.sh（可并行）。
#
#   setsid nohup bash /data1/tau0wm/local_scripts/h100/relay_00091_after_00073.sh \
#       > /data1/tau0wm/relay_00091.out 2>&1 &
#   tail -f /data1/tau0wm/relay_00091.log
set -uo pipefail

ROOT="${TAU_H100_ROOT:-/data1/tau0wm}"
S="${ROOT}/local_scripts"
LOG="${ROOT}/relay_00091.log"
RUNLOG="${ROOT}/logs"
GPUS="${GPUS:-2,3,4,5,6,7}"
SMOKE_GPU="${SMOKE_GPU:-2}"
MEM_FREE_MIB="${MEM_FREE_MIB:-3000}"
NEED_STABLE="${NEED_STABLE:-3}"
POLL="${POLL:-120}"
MAX_WAIT_H="${MAX_WAIT_H:-24}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-5400}"
STARTUP_WAIT="${STARTUP_WAIT:-2400}"
CONFIG_YAML="${S}/configs/tau_model/posttrain_tongbot_00091_ee_300k.yaml"
TRAINLOG="${RUNLOG}/train_00091_300k_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "${RUNLOG}"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }
fatal() { log "RELAY_FATAL: $*"; exit "${2:-1}"; }

# shellcheck disable=SC1091
[[ -f "${S}/h100/gwp_gate.sh" ]] || fatal "缺 gwp_gate.sh"
source "${S}/h100/gwp_gate.sh"

still_00073() {
  pgrep -f "posttrain_tongbot_00073" >/dev/null 2>&1
}

peak_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPUS}" 2>/dev/null \
    | sort -n | tail -1
}

preflight() {
  log "== 前置：权重 + 补丁 =="
  [[ -f "${ROOT}/pretrained/tau-0-wm/config.json" ]] || fatal "缺 VAM 权重"
  [[ -f "${CONFIG_YAML}" ]] || fatal "缺 ${CONFIG_YAML}"
  "${ROOT}/envs/tau0wm/bin/python" "${S}/patch_tau_repo.py" --repo "${ROOT}/repo/tau-0-wm" \
    >>"${LOG}" 2>&1 || fatal "patch_tau_repo.py 失败"
  log "前置通过"
}

data_ready() {
  [[ -f "${S}/configs/stat_file/tongbot_00091_ee_rela.json" ]] \
    && [[ -f "${ROOT}/datasets/00091_rm75_ee_tau0wm/meta/info.json" ]] \
    && grep -q '"codebase_version": "v3.0"' "${ROOT}/datasets/00091_rm75_ee_tau0wm/meta/info.json"
}

wait_convert() {
  local deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
  log "== 等 00091 EE 转换完成 =="
  while :; do
    if data_ready; then
      log "00091 EE v3.0 + 统计已齐"; return 0
    fi
    if (( $(date +%s) > deadline )); then
      fatal "等转换超过 ${MAX_WAIT_H}h"
    fi
    if pgrep -f "convert_00091_ee_on_h100.sh" >/dev/null 2>&1; then
      log "转换还在跑，继续等"
    else
      fatal "转换没在跑且数据不齐，见 ${ROOT}/convert_00091.log"
    fi
    sleep "${POLL}"
  done
}

wait_00073() {
  local deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
  local stable=0 n
  log "== 等 00073 结束（GPU ${GPUS}，每 ${POLL}s，上限 ${MAX_WAIT_H}h）=="
  while :; do
    if [[ -f "${ROOT}/HOLD" ]]; then
      log "HOLD 存在，暂停"; sleep "${POLL}"; stable=0; continue
    fi
    if (( $(date +%s) > deadline )); then
      fatal "等待超过 ${MAX_WAIT_H}h" 2
    fi
    if still_00073; then
      (( stable != 0 )) && log "00073 还在，重新计时"
      stable=0
      sleep "${POLL}"
      continue
    fi
    n="$(peak_used)"
    [[ -n "${n}" ]] || fatal "nvidia-smi 失败"
    if (( n >= MEM_FREE_MIB )); then
      stable=0
      log "00073 进程已退，但 GPU 峰值仍 ${n} MiB → 继续等"
      sleep "${POLL}"
      continue
    fi
    stable=$(( stable + 1 ))
    log "00073 已退、GPU 峰值 ${n} MiB（${stable}/${NEED_STABLE}）"
    (( stable >= NEED_STABLE )) && { log "放行：GPU ${GPUS} 空闲"; return 0; }
    sleep "${POLL}"
  done
}

smoke() {
  log "== 单卡冒烟 GPU ${SMOKE_GPU} =="
  local out="${RUNLOG}/smoke_00091_$(date '+%Y%m%d_%H%M%S').log"
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" CONFIG="${CONFIG_YAML}" \
    timeout "${SMOKE_TIMEOUT}" bash "${S}/launch_train.sh" --smoke >"${out}" 2>&1
  local rc=$?
  log "冒烟退出码 ${rc}，日志 ${out}"
  (( rc == 0 )) || { tail -40 "${out}" | tee -a "${LOG}"; fatal "冒烟失败 rc=${rc}"; }
  grep -aq "loss=" "${out}" || fatal "冒烟没有 loss="
  grep -aqE "CUDA out of memory|Traceback" "${out}" && fatal "冒烟有异常"
  log "冒烟通过"
}

formal() {
  log "== 六卡正式 00091 300k，日志 ${TRAINLOG} =="
  sleep 45
  local n; n="$(peak_used)"
  (( n < MEM_FREE_MIB )) || fatal "冒烟后退卡仍 ${n} MiB"
  # shellcheck disable=SC1091
  source "${S}/h100/env_h100.sh" >>"${LOG}" 2>&1
  export CONFIG="${CONFIG_YAML}"
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  setsid nohup bash "${S}/launch_train.sh" >"${TRAINLOG}" 2>&1 < /dev/null &
  local waited=0
  while (( waited < STARTUP_WAIT )); do
    sleep 60; waited=$(( waited + 60 ))
    if tr '\r' '\n' < "${TRAINLOG}" 2>/dev/null | grep -aq 'loss='; then
      tr '\r' '\n' < "${TRAINLOG}" | grep -a 'loss=' | tail -3 | tee -a "${LOG}"
      log "RELAY_DONE 00091 六卡已出 loss（${waited}s）log=${TRAINLOG}"
      return 0
    fi
    if grep -aqE "CUDA out of memory|Traceback" "${TRAINLOG}" 2>/dev/null; then
      tail -60 "${TRAINLOG}" | tee -a "${LOG}"
      fatal "00091 早期异常" 3
    fi
    if ! pgrep -f "posttrain_tongbot_00091_ee_300k" >/dev/null 2>&1; then
      tail -60 "${TRAINLOG}" | tee -a "${LOG}"
      fatal "00091 进程消失" 3
    fi
    (( waited % 300 == 0 )) && log "启动中…${waited}s"
  done
  fatal "${STARTUP_WAIT}s 内没有 loss" 3
}

: > "${LOG}"
log "watcher pid=$$ 起（等 00073 → 训 00091 GPU ${GPUS}）"
preflight
wait_00073
wait_convert
smoke
formal
