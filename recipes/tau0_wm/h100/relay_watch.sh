#!/usr/bin/env bash
# τ0-WM 自动接力：等 h100-103 上的 GigaWorld-Policy 训练收口、GPU 2-7 空出来，
# 再自动跑「单卡冒烟 → 六卡正式训练」。
#
# 用户硬要求：**GWP 没跑完之前绝不启动 τ0-WM**。本脚本因此不做任何「抢卡」动作，
# 只在 GPU 2-7 显存真正掉下来之后才动。判定用 h100/gwp_gate.sh 的「GWP 家族」
# （训练进程 + run_task 入口 + 会把它重新拉起的看门狗），避免 bs4→bs8 那种
# kill-再重启 的几分钟空窗被误读成「GWP 结束了」。
#
# 部署（在 h100 上，必须 setsid nohup 后台化，ssh 长会话会被网关 ~4min 掐断）：
#   setsid nohup bash /data1/tau0wm/local_scripts/h100/relay_watch.sh \
#       > /data1/tau0wm/relay.out 2>&1 &
#
# 人工闸门：
#   touch /data1/tau0wm/HOLD     # 让 watcher 停在「等待」阶段不往下走（已启动的训练不受影响）
#   看进度：tail -f /data1/tau0wm/relay.log
# 终态标记：relay.log 末尾 RELAY_DONE / RELAY_FATAL
set -uo pipefail

ROOT="${TAU_H100_ROOT:-/data1/tau0wm}"
S="${ROOT}/local_scripts"
LOG="${ROOT}/relay.log"
RUNLOG="${ROOT}/logs"
GPUS="${GPUS:-2,3,4,5,6,7}"          # 只吃 GWP 腾出来的这 6 张，0/1 是别人的
SMOKE_GPU="${SMOKE_GPU:-2}"
MEM_FREE_MIB="${MEM_FREE_MIB:-3000}" # 单卡 used 低于此值才算空（桌面/残留进程占 ~1-2G）
NEED_STABLE="${NEED_STABLE:-3}"      # 连续 N 次判定「GWP 已退且卡已空」才放行，防抖动
POLL="${POLL:-120}"
MAX_WAIT_H="${MAX_WAIT_H:-48}"       # 等 GWP 的上限，超时退出等人
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-5400}"
STARTUP_WAIT="${STARTUP_WAIT:-2400}"   # 正式训练从拉起到看见第一条 loss 的容忍上限
DATASET="${ROOT}/datasets/00036_rm75_ee_tau0wm"

mkdir -p "${RUNLOG}"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }
fatal() { log "RELAY_FATAL: $*"; exit "${2:-1}"; }

# GWP 门控必须在任何判定之前就位：source 失败还继续跑，
# gwp_running 会因函数未定义而返回非零 = 「GWP 不在跑」= 误放行抢卡。
# shellcheck disable=SC1091
[[ -f "${S}/h100/gwp_gate.sh" ]] || { log "缺 ${S}/h100/gwp_gate.sh，拒绝启动 watcher"; exit 1; }
source "${S}/h100/gwp_gate.sh"
gwp_running() { gwp_family_running; }

ts() { date '+%Y%m%d_%H%M%S'; }

# ---------------------------------------------------------------- 阶段 0：前置自检
# 宁可现在 FATAL，也别等 GWP 跑完（可能是一觉醒来）才发现少东西
preflight() {
  log "== 阶段 0 前置自检 =="
  grep -q "SETUP_DONE" "${ROOT}/setup_h100.log" 2>/dev/null \
    || fatal "setup_h100.log 里没有 SETUP_DONE（环境还没装完）"
  [[ -f "${ROOT}/pretrained/tau-0-wm/config.json" ]] \
    || fatal "缺 VAM 权重 ${ROOT}/pretrained/tau-0-wm/config.json"
  [[ -f "${ROOT}/pretrained/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" ]] \
    || fatal "缺 Wan2.2 VAE（setup_h100.sh 第 5 步的软链没做成）"
  [[ -f "${ROOT}/pretrained/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" ]] \
    || fatal "缺 umt5-xxl T5"
  [[ -d "${ROOT}/pretrained/Wan2.2-TI2V-5B/google/umt5-xxl" ]] \
    || fatal "缺 umt5-xxl tokenizer"
  [[ -f "${DATASET}/meta/info.json" ]] || fatal "缺数据集 ${DATASET}"
  [[ -f "${S}/configs/stat_file/tongbot_00036_ee_rela.json" ]] || fatal "缺 EE 统计文件"
  [[ -f "${ROOT}/repo/tau-0-wm/main.py" ]] || fatal "缺 τ0-WM 仓库 ${ROOT}/repo/tau-0-wm"

  # fork 补丁（上游 forward_pass 死导入），幂等
  "${ROOT}/envs/tau0wm/bin/python" "${S}/patch_tau_repo.py" --repo "${ROOT}/repo/tau-0-wm" \
    >>"${LOG}" 2>&1 || fatal "patch_tau_repo.py 失败"
  # 数据 YAML 里的数据集前缀必须是本机路径
  grep -q "data_roots: \[\"${DATASET}\"\]" "${S}/configs/data/tongbot_00036_ee_rela.yaml" \
    || log "WARN：data YAML 的 data_roots 与 ${DATASET} 不一致，检查同步是否漏了"
  # 门控得真的「看得见」GWP 才算武装好：此刻若在跑，必须能列出进程；
  # 若启动时就看不见、GPU 又还占着，说明匹配串过时了 → 宁可直接失败也不要误放行。
  if gwp_family_running; then
    log "门控已武装，当前 GWP 家族进程："; gwp_family_desc | tee -a "${LOG}"
  elif (( $(peak_used) >= MEM_FREE_MIB )); then
    fatal "GPU ${GPUS} 峰值仍 $(peak_used) MiB，但 GWP 家族一个进程都没匹配上（匹配串过时？人工确认机器在跑什么再重启 watcher）"
  else
    log "WARN：启动时 GWP 未在跑且 GPU 空闲，将直接放行（确认这是你要的）"
  fi
  log "前置自检通过"
}

# ---------------------------------------------------------------- 阶段 1：等 GWP
# GPU 2-7 里 used 最大的那张
peak_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPUS}" 2>/dev/null \
    | sort -n | tail -1
}

# GPU 2-7 是否已经空出来（连续 NEED_STABLE 次判定）
assert_gpus_free() {
  local need="${1:-${NEED_STABLE}}" stable=0 n
  while :; do
    if [[ -f "${ROOT}/HOLD" ]]; then
      log "HOLD 存在，暂停接力"; sleep "${POLL}"; stable=0; continue
    fi
    if (( $(date +%s) > DEADLINE )); then
      fatal "等待超过 ${MAX_WAIT_H}h，放弃（人工确认后再重启 watcher）" 2
    fi
    if gwp_running; then
      (( stable != 0 )) && log "GWP 又出现了，重新计时"
      stable=0; sleep "${POLL}"; continue
    fi
    n="$(peak_used)"
    [[ -n "${n}" ]] || fatal "nvidia-smi -i ${GPUS} 查不到显存"
    if (( n >= MEM_FREE_MIB )); then
      stable=0
      log "GWP 进程已退，但 GPU ${GPUS} 峰值仍 ${n} MiB（≥${MEM_FREE_MIB}）→ 继续等"
      sleep "${POLL}"; continue
    fi
    stable=$(( stable + 1 ))
    log "GWP 已退、GPU 峰值 ${n} MiB（${stable}/${need}）"
    (( stable >= need )) && { log "放行：GPU ${GPUS} 空闲"; return 0; }
    sleep "${POLL}"
  done
}

wait_gwp() {
  log "== 阶段 1 等 GWP 结束（GPU ${GPUS}，每 ${POLL}s 查，上限 ${MAX_WAIT_H}h）=="
  assert_gpus_free "${NEED_STABLE}"
}


# ---------------------------------------------------------------- 阶段 2：冒烟
smoke() {
  log "== 阶段 2 单卡冒烟（GPU ${SMOKE_GPU}，bs=1 / 20 step，上限 ${SMOKE_TIMEOUT}s）=="
  local out="${RUNLOG}/smoke_$(ts).log" peak_pid="" peak_file="${RUNLOG}/smoke_peak.txt"
  # 顺手记 GPU 峰值显存，给「正式训练能不能把 bs 从 8 往上提」提供依据
  ( : > "${peak_file}"
    while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                   -i "${SMOKE_GPU}" >>"${peak_file}" 2>/dev/null; sleep 5; done ) &
  peak_pid=$!

  CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" timeout "${SMOKE_TIMEOUT}" \
    bash "${S}/launch_train.sh" --smoke >"${out}" 2>&1
  local rc=$?
  kill "${peak_pid}" 2>/dev/null; wait "${peak_pid}" 2>/dev/null
  local peak; peak=$(sort -n "${peak_file}" 2>/dev/null | tail -1)
  log "冒烟退出码 ${rc}，GPU${SMOKE_GPU} 峰值显存 ${peak:-?} MiB，日志 ${out}"

  (( rc == 0 )) || { tail -40 "${out}" | tee -a "${LOG}"; fatal "冒烟退出码 ${rc}（124=超时）"; }
  # loss 走 tqdm postfix（stderr，\r 分隔），有 loss= 才算真的跑起来了
  grep -aq "loss=" "${out}" || { tail -40 "${out}" | tee -a "${LOG}"; fatal "冒烟日志里没有 loss="; }
  grep -aqE "CUDA out of memory|Traceback|Error action value" "${out}" \
    && fatal "冒烟日志里有异常，见 ${out}"
  log "冒烟通过：$(tr '\r' '\n' < "${out}" | grep -a 'loss=' | tail -1)"
}

# ---------------------------------------------------------------- 阶段 3：正式
formal() {
  log "== 阶段 3 六卡正式训练（GPU ${GPUS}，30000 step，日志 ${TRAINLOG}）=="
  # 冒烟期间可能已被别人占走，放行条件再查一次（只要 GPU 空就继续）
  sleep 45                      # 等冒烟进程的显存真正归还
  assert_gpus_free 1
  CUDA_VISIBLE_DEVICES="${GPUS}" setsid nohup bash "${S}/launch_train.sh" \
      >"${TRAINLOG}" 2>&1 &
  local waited=0
  # 六进程各要加载 22.9G VAM + T5 + VAE，首个 step 前给足时间
  while (( waited < STARTUP_WAIT )); do
    sleep 60; waited=$(( waited + 60 ))
    if tr '\r' '\n' < "${TRAINLOG}" 2>/dev/null | grep -aq 'loss='; then
      tr '\r' '\n' < "${TRAINLOG}" | grep -a 'loss=' | tail -3 | tee -a "${LOG}"
      log "RELAY_DONE τ0-WM 六卡训练已出 loss（等待 ${waited}s），log=${TRAINLOG}"
      return 0
    fi
    if grep -aqE "CUDA out of memory|Traceback|Error action value" "${TRAINLOG}" 2>/dev/null; then
      tail -60 "${TRAINLOG}" | tee -a "${LOG}"
      fatal "正式训练早期就报异常，见 ${TRAINLOG}" 3
    fi
    if ! pgrep -f "posttrain_tongbot_00036_ee" >/dev/null 2>&1; then
      tail -60 "${TRAINLOG}" | tee -a "${LOG}"
      fatal "启动 ${waited}s 后训练进程已消失，见 ${TRAINLOG}" 3
    fi
    (( waited % 300 == 0 )) && log "启动中…${waited}s（权重加载 / 首个 batch）"
  done
  fatal "${STARTUP_WAIT}s 内没有 step 日志（可能卡在 dataloader / 首个 batch），见 ${TRAINLOG}" 3
}

# ---------------------------------------------------------------- 主流程
DEADLINE=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
preflight
log "watcher pid=$$ 起（GWP 门控：train='${GWP_TRAIN_MATCH}' / task='${GWP_TASK_MATCH}' / watchdog='${GWP_WATCH_MATCH}'）"
# shellcheck disable=SC1091
source "${S}/h100/env_h100.sh" >>"${LOG}" 2>&1
TRAINLOG="${RUNLOG}/train_$(ts).log"
wait_gwp
smoke
formal

