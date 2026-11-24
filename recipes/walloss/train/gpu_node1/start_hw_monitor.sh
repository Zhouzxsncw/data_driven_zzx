#!/usr/bin/env bash
# 训练 sidecar：采 GPU/PPU/NPU 利用率+显存，不碰训练进程。
# 用法：. 本文件 然后 start_hw_monitor "$LOG"
# 关：HW_MONITOR=0 ；间隔：HW_MONITOR_INTERVAL=30
start_hw_monitor() {
  local log="${1:-}"
  local devices="${2:-${CUDA_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-}}}"
  local watch_pid="${3:-${HW_MONITOR_PID:-$$}}"
  local interval="${HW_MONITOR_INTERVAL:-30}"
  [ "${HW_MONITOR:-1}" = "0" ] && return 0

  local here py d out err
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd 2>/dev/null || echo .)"
  py=""
  for d in \
    "$here" \
    "$here/.." \
    /home/xk/data_Driven/scripts \
    /home/pi05/openpi_src \
    /home/pi05/openpi_src/scripts \
    /workspace/openpi/cigai_train/openpi-jiangsuan/scripts \
    /home/ubuntu/cigai_train/openpi-jiangsuan/scripts \
    /workspace/code/train \
    /nvme2n1/wall-oss/code/train \
    /data/wall-oss/code/train \
    /data1/hw_monitor
  do
    if [ -f "$d/hw_train_monitor.py" ]; then
      py="$d/hw_train_monitor.py"
      break
    fi
  done
  if [ -z "$py" ]; then
    echo "[hw-monitor] skip: hw_train_monitor.py not found"
    return 0
  fi

  if [ -n "${LOGDIR:-}" ]; then
    out="${HW_MONITOR_OUT:-${LOGDIR}/hw.jsonl}"
  elif [ -n "$log" ]; then
    out="${HW_MONITOR_OUT:-${log}.hw.jsonl}"
  else
    out="${HW_MONITOR_OUT:-./train.hw.jsonl}"
  fi
  mkdir -p "$(dirname "$out")" 2>/dev/null || true
  err="${out%.jsonl}.err"
  export PATH="/usr/local/PPU_SDK/ppu-smi/bin:${PATH:-}"

  local pybin=""
  if command -v python3 >/dev/null 2>&1; then
    pybin=python3
  elif command -v python >/dev/null 2>&1; then
    pybin=python
  else
    echo "[hw-monitor] skip: no python3"
    return 0
  fi

  # setsid：exec 换成 torchrun 后不被 SIGHUP 带走，才能写出 summary
  setsid "$pybin" "$py" \
    --out "$out" \
    --interval "$interval" \
    --pid "$watch_pid" \
    --devices "${devices:-}" \
    --wait-start 120 \
    </dev/null >/dev/null 2>>"$err" &
  echo "[hw-monitor] sidecar=$! watch_pid=$watch_pid out=$out interval=${interval}s devices=${devices:-all} py=$py"
}

if [ "${BASH_SOURCE[0]-}" = "$0" ]; then
  start_hw_monitor "${1:-}" "${2:-}" "${3:-}"
fi
