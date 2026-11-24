#!/usr/bin/env bash
# FastWAM 续训期间的内存采样器：把 resume 峰值/存盘峰值留成硬数据（两次 OOM 都是靠这个定位的）。
# 用法：setsid nohup bash mem_probe.sh >/dev/null 2>&1 &     INTERVAL=20 LOG=<path> bash mem_probe.sh
set -uo pipefail

LOG="${LOG:-/data/FastWAM_runs/tongbot_00036/logs/mem_probe.log}"
INTERVAL="${INTERVAL:-20}"

# 真正的 trainer 是 cmdline 含 scripts/train.py 且 comm 为 python 的那个（accelerate 启动器也含该串）
trainer_pid() {
  local p
  for p in $(pgrep -f "scripts/train.py" 2>/dev/null); do
    [[ "$(cat "/proc/${p}/comm" 2>/dev/null)" == python* ]] || continue
    tr '\0' ' ' < "/proc/${p}/cmdline" 2>/dev/null | grep -q "output_dir=" && { echo "${p}"; return; }
  done
  echo ""
}

mkdir -p "$(dirname "${LOG}")" 2>/dev/null || true
printf '%s START probe interval=%ss\n' "$(date -Is)" "${INTERVAL}" >>"${LOG}"
while true; do
  tp="$(trainer_pid)"
  prss=0; psw=0; wrss=0; wn=0; wpss=0; wpriv=0
  if [[ -n "${tp}" ]]; then
    read -r prss psw < <(awk '/VmRSS/{r=int($2/1048576)}/VmSwap/{s=int($2/1048576)}END{print r+0, s+0}' "/proc/${tp}/status" 2>/dev/null)
    # dataloader worker 的 RSS 里含与父进程 COW 共享的页，看总量趋势而非绝对值
  # 注意：worker 的 cmdline 与父进程一模一样，只能按 comm 精确匹配（pgrep -x）
    wpss=0; wpriv=0
    for w in $(pgrep -x pt_data_worker 2>/dev/null); do
      v="$(awk '/VmRSS/{print int($2/1048576)}' "/proc/${w}/status" 2>/dev/null)"
      [[ -n "${v}" ]] || continue
      wrss=$((wrss + v)); wn=$((wn + 1))
      # RSS 会把与父进程 COW 共享的页重复计数；Pss 按共享者均摊、Private_Dirty 才是 worker 真正独占的
      read -r pss priv < <(awk '/^Pss:/{p=$2}/^Private_Dirty:/{d=$2}END{printf "%d %d", p/1048576, d/1048576}' "/proc/${w}/smaps_rollup" 2>/dev/null)
      wpss=$((wpss + ${pss:-0})); wpriv=$((wpriv + ${priv:-0}))
    done
  fi
  # machine_anon = 全机 anon 物理占用（active+inactive），OOM 前它 + swap 会逼近 MemTotal+SwapTotal；
  # 这个数才是真相，单进程 RSS 会因为 fork 后的 COW 共享页被重复计数。
  manon="$(awk '/^Active\(anon\)|^Inactive\(anon\)/{s+=$2}END{printf "%d", s/1048576}' /proc/meminfo)"
  printf '%s avail=%sG swapfree=%sG machine_anon=%sG sys_shmem=%sG train_rss=%sG train_swap=%sG workers=%s worker_rss_sum=%sG worker_pss=%sG worker_priv_dirty=%sG pid=%s\n' \
    "$(date +%H:%M:%S)" \
    "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" \
    "$(awk '/SwapFree/{print int($2/1048576)}' /proc/meminfo)" \
    "${manon}" \
    "$(awk '/^Shmem:/{print int($2/1048576)}' /proc/meminfo)" \
    "${prss}" "${psw}" "${wn}" "${wrss}" "${wpss:-0}" "${wpriv:-0}" "${tp:-dead}" >>"${LOG}"
  sleep "${INTERVAL}"
done
