#!/usr/bin/env bash
# FastWAM 续训期间的 accelerate state 轮转：只保留最近 KEEP 个「完整」state。
#
# 为什么必须轮转：一份 state ≈80G（72G optimizer + 13G model），save_every=2500 → 本次续训会连存
# 12500/15000/17500/20000 共 4 份 = 320G，而 /data 只剩 ~154G；不轮转会在第二份存盘时写爆，
# 把训练进程连同半成品 state 一起打死。
#
# 安全边界（四条同时满足才删）：
#   1) 目标目录严格匹配 ^step_[0-9]{6}$ 且就在 STATE_ROOT 之下（不递归、不跟其它路径）
#   2) 完整性判据：trainer_state.json 存在（它在 accelerator.save_state 全部落盘后才由主进程写）
#      且 pytorch_model/mp_rank_00_model_states.pt > 1G
#   3) 永远保留按步数排序最新的 KEEP 个；被删者步数必须严格小于保留者
#   4) rm 带 --one-file-system，且先把体积/时间写进日志
#
# 用法：
#   ONESHOT=1 DRY_RUN=1 bash keep_last_state.sh     # 看会删什么，不动手
#   bash keep_last_state.sh                          # 后台常驻，每 INTERVAL 秒一轮
set -uo pipefail

STATE_ROOT="${STATE_ROOT:-/data/FastWAM_runs/tongbot_00036/checkpoints/state}"
KEEP="${KEEP:-1}"
STALE_SECS="${STALE_SECS:-1800}"   # 半成品目录多久没被写过才算「已废弃」
INTERVAL="${INTERVAL:-300}"
LOG="${LOG:-/data/FastWAM_runs/tongbot_00036/logs/keep_last.log}"
ONESHOT="${ONESHOT:-0}"
DRY_RUN="${DRY_RUN:-0}"

say() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "${LOG}" 2>/dev/null || printf '%s %s\n' "$(date -Is)" "$*"; }

is_complete() {
  local d="$1"
  [[ -f "${d}/trainer_state.json" ]] || return 1
  local sz
  sz="$(stat -c %s "${d}/pytorch_model/mp_rank_00_model_states.pt" 2>/dev/null || echo 0)"
  [[ "${sz}" -gt 1000000000 ]] || return 1
  ls "${d}"/pytorch_model/*optim_states.pt >/dev/null 2>&1 || return 1
  return 0
}

pass() {
  local complete=() d n victim newest size name num num_newest mtime now
  while IFS= read -r d; do [[ -n "$d" ]] && complete+=("$d"); done < <(
    for cand in "${STATE_ROOT}"/step_[0-9][0-9][0-9][0-9][0-9][0-9]; do
      [[ -d "${cand}" ]] || continue
      [[ "$(basename "${cand}")" =~ ^step_[0-9]{6}$ ]] || continue
      is_complete "${cand}" && basename "${cand}"
    done | sort -V
  )
  n=${#complete[@]}
  if (( n == 0 )); then say "OK   无完整 state（跳过轮转）"; return 0; fi
  newest="${complete[$((n-1))]}"
  num_newest="$(tr -dc '0-9' <<<"${newest}")"

  # 先处理「半成品」目录：被 SIGKILL/OOM 打断的 save_state 会留下无 trainer_state.json 的目录，
  # 它不在 complete 列表里，永远轮转不到 → 80G 白占磁盘，会把下一次存盘写爆。
  # 删除条件（三条同时满足）：步数严格小于最新完整 state（说明它已被更新的一份取代）
  #                        且目录内文件已 STALE_SECS 秒没被写过（说明不在落盘中）
  #                        且名字严格合法。步数 >= 最新完整者只 WARN，绝不删（多半是正在写的份）。
  for d in "${STATE_ROOT}"/step_[0-9][0-9][0-9][0-9][0-9][0-9]; do
    [[ -d "${d}" ]] || continue
    name="$(basename "${d}")"
    [[ "${name}" =~ ^step_[0-9]{6}$ ]] || continue
    is_complete "${d}" && continue
    num="$(tr -dc '0-9' <<<"${name}")"
    # 注意：目录名带前导零，必须用 10# 强制十进制，否则 (( )) 按八进制解析直接报错
    if (( 10#${num} >= 10#${num_newest} )); then
      say "WARN 未完成但不比最新完整 state(${newest}) 旧：${name}，可能正在写入，不处理"
      continue
    fi
    mtime="$(find "${d}" -type f -printf '%T@\n' 2>/dev/null | cut -d. -f1 | sort -rn | head -1)"
    now="$(date +%s)"
    if [[ -z "${mtime}" ]] || (( now - mtime < STALE_SECS )); then
      say "SKIP ${name} 未完成且 ${mtime:-0}s 前仍在写，暂不删"
      continue
    fi
    size="$(du -sh "${d}" 2>/dev/null | cut -f1)"
    if [[ "${DRY_RUN}" == "1" ]]; then
      say "DRY  会删半成品 ${STATE_ROOT}/${name} (${size})"
    else
      say "PRUNE-STALE ${STATE_ROOT}/${name} (${size}) 未完成、$(( (now - mtime) / 60 ))min 无写入、已保留 ${newest}"
      rm -rf --one-file-system "${d}" \
        && say "DONE 已删半成品 ${name}，free=$(df -BG --output=avail "${STATE_ROOT}" | tail -1 | tr -dc '0-9')G" \
        || say "FAIL rm 半成品非零退出：${name}"
    fi
  done

  if (( n <= KEEP )); then
    say "OK   complete=${n} keep=${KEEP}（无可剪项）free=$(df -BG --output=avail "${STATE_ROOT}" | tail -1 | tr -dc '0-9')G"
    return 0
  fi
  for victim in "${complete[@]:0:n-KEEP}"; do
    [[ "${victim}" =~ ^step_[0-9]{6}$ ]] || { say "SKIP 非法目录名 ${victim}"; continue; }
    [[ "${victim}" < "${newest}" || 10#"$(tr -dc '0-9' <<<"${victim}")" -lt 10#"${num_newest}" ]] \
      || { say "SKIP ${victim} 不比最新 ${newest} 旧"; continue; }
    size="$(du -sh "${STATE_ROOT}/${victim}" 2>/dev/null | cut -f1)"
    if [[ "${DRY_RUN}" == "1" ]]; then
      say "DRY  会删 ${STATE_ROOT}/${victim} (${size})，保留最新 ${newest}"
    else
      say "PRUNE ${STATE_ROOT}/${victim} (${size}) → 保留最新 ${newest}"
      rm -rf --one-file-system "${STATE_ROOT}/${victim}" \
        && say "DONE 已删 ${victim}，free=$(df -BG --output=avail "${STATE_ROOT}" | tail -1 | tr -dc '0-9')G" \
        || say "FAIL rm 非零退出：${victim}"
    fi
  done
}

mkdir -p "$(dirname "${LOG}")" 2>/dev/null || true
[[ -d "${STATE_ROOT}" ]] || { say "FAIL state 根目录不存在：${STATE_ROOT}"; exit 1; }
if [[ "${ONESHOT}" == "1" ]]; then pass; exit 0; fi
say "START state_root=${STATE_ROOT} keep=${KEEP} interval=${INTERVAL}s dry=${DRY_RUN}"
while true; do
  pass
  sleep "${INTERVAL}"
done
