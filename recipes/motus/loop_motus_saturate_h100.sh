#!/usr/bin/env bash
# ============================================================
# h100-103 满载接力：等 DreamZero-14B 让出 8 卡 → 立刻开 Motus-00091 正式训练
#
# 目的：机器现在被 DreamZero-14B 100% 占满（8 卡 × 79.5G），Motus 的栈也在这台机上
# 装完但一步没跑。这个脚本让两张卡的时间线首尾相接，中间不落空：
#   DreamZero 跑完/挂掉 → 卡空闲 2 个采样周期 → 拉起 launch_motus_00091_h100.sh MODE=full
#
# 用法（在 H100 上后台化）：
#   mkdir -p /data/train_data/motus-outputs/tongbot_00091
#   setsid nohup bash /data1/motus/recipes/loop_motus_saturate_h100.sh \
#     > /data/train_data/motus-outputs/tongbot_00091/saturate.out 2>&1 < /dev/null &
#
# 可覆盖：POLL / NEED_FREE / STABLE_ROUNDS / MAX_WAIT_H / OUT_ROOT / DZ_LOG
# 只读等待阶段不碰 GPU、不影响 DreamZero。
# ============================================================
set -uo pipefail

RECIPE_DIR="${RECIPE_DIR:-/data1/motus/recipes}"
LAUNCHER="${LAUNCHER:-${RECIPE_DIR}/launch_motus_00091_h100.sh}"
OUT_ROOT="${OUT_ROOT:-/data/train_data/motus-outputs}"
SAT_DIR="${OUT_ROOT}/tongbot_00091"
SAT_LOG="${SAT_DIR}/saturate.log"
LOCK="${SAT_DIR}/.saturate.lock"

NPROC="${NPROC:-8}"
POLL="${POLL:-60}"                       # 秒
NEED_FREE="${NEED_FREE:-3000}"           # 单卡占用低于此值算空闲（MiB）
STABLE_ROUNDS="${STABLE_ROUNDS:-2}"      # 连续 N 次空闲才交接，避开瞬时抖动
MAX_WAIT_H="${MAX_WAIT_H:-12}"           # 最长等待（小时），超时放弃，避免无限挂着
DZ_LOG="${DZ_LOG:-/data/train_data/dreamzero-outputs/tong00073_sft_dreamzero_14b/full_full20260910b/sft/train.log}"

mkdir -p "${SAT_DIR}" || { echo "写不了 ${SAT_DIR}"; exit 1; }
if ! mkdir "${LOCK}" 2>/dev/null; then
  # 已有接力在跑：报一声就退出，避免两个 torchrun 抢同一组卡
  echo "[$(date '+%F %T')] 已有 saturate 接力在跑（${LOCK} 存在），本次退出" | tee -a "${SAT_LOG}"
  exit 0
fi
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

log() { echo "[$(date '+%F %T')] $*" | tee -a "${SAT_LOG}"; }

log "========== Motus-00091 满载接力启动 =========="
log "等待条件：${NPROC} 卡全部 < ${NEED_FREE}MiB 且无 train_vla_sft/train.py 进程，连续 ${STABLE_ROUNDS} 次"
log "交接动作：MODE=full bash ${LAUNCHER}"

[ -f "${LAUNCHER}" ] || { log "FATAL 缺 ${LAUNCHER}"; exit 1; }

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
free_rounds=0
last_report=0

while :; do
  now=$(date +%s)
  used_sum=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc)
  used_sum=${used_sum:-999999}
  busy_procs=$(pgrep -fc 'train_vla_sft|train/train\.py' 2>/dev/null | head -1)
  busy_procs=${busy_procs:-0}

  if [ "${used_sum}" -lt $(( NEED_FREE * NPROC )) ] && [ "${busy_procs}" -eq 0 ]; then
    free_rounds=$(( free_rounds + 1 ))
    log "卡空闲 ${free_rounds}/${STABLE_ROUNDS}（合计 ${used_sum}MiB，训练进程 ${busy_procs}）"
    if [ "${free_rounds}" -ge "${STABLE_ROUNDS}" ]; then
      # 交接前把上一棒的结果记一行，方便事后复盘
      if [ -f "${DZ_LOG}" ]; then
        log "上一棒 DreamZero 末态：$(tr '\r' '\n' < "${DZ_LOG}" | grep -Eo 'Global Step:.*[0-9]+/6000 \[[^]]*\]' | tail -1)"
      fi
      log "========== 交接：开 Motus-00091 正式训练 =========="
      MODE=full STAMP=relay_$(date +%Y%m%d-%H%M) bash "${LAUNCHER}" 2>&1 | tee -a "${SAT_LOG}"
      rc=${PIPESTATUS[0]}
      log "Motus-00091 启动器退出 rc=${rc}；接力结束（要再开一次请重跑本脚本）"
      exit "${rc}"
    fi
  else
    free_rounds=0
    # 每 10 分钟才打一条“还在等”，避免日志被刷
    if [ $(( now - last_report )) -ge 600 ]; then
      last_report=${now}
      log "仍在等待：合计 ${used_sum}MiB / 训练进程 ${busy_procs}（DreamZero 未完）"
    fi
  fi

  if [ "${now}" -gt "${deadline}" ]; then
    log "等待超过 ${MAX_WAIT_H}h，放弃接力（卡仍被占：${used_sum}MiB）"
    exit 3
  fi
  sleep "${POLL}"
done
