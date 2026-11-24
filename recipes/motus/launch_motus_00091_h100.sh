#!/usr/bin/env bash
# ============================================================
# h100-103 上运行：Motus 5B (Wan2.2-TI2V) TongBot 00091 抽屉药盒 SFT
# 8x H100 80GB，DeepSpeed ZeRO-1 + bf16，per-card bs=4 → 全局 32
#
# 用法（在 H100 上）:
#   MODE=check bash /data1/motus/recipes/launch_motus_00091_h100.sh    # 只做 preflight（不占卡）
#   bash        /data1/motus/recipes/launch_motus_00091_h100.sh        # preflight + 20 步冒烟
#   MODE=full  bash /data1/motus/recipes/launch_motus_00091_h100.sh    # 冒烟→正式（崩溃自愈 + keep-last-N）
#   MODE=full SKIP_SMOKE=1 bash .../launch_motus_00091_h100.sh         # 卡已确认能跑，跳过冒烟
#
# 可覆盖: OUT_ROOT / BS / MAX_STEPS / SAVE_INTERVAL / KEEP_CKPTS / NUM_WORKERS / NPROC /
#         SKIP_PREFLIGHT / SKIP_SMOKE / MAX_RETRIES
#
# 设计约束（都是踩过的坑）：
#   * ckpt 落点必须离开 /data1（只剩 ~366G）→ 默认 /data/train_data
#   * --config 的 basename 会被 train.py 当 dataset_name 拼进 ckpt 路径，
#     所以运行时 yaml 必须仍叫 tongbot_00091.yaml（放 configs_run/）
#   * max_steps / save_interval 只能走 yaml，train.py 没有对应 CLI
#   * 训练结束时 rank0 会无条件再存一次 → 冒烟也要立刻清盘
#   * accelerator.save_state 同时写 DeepSpeed 分片 + rank0 的 pytorch_model_0.bin，
#     单份 60~80G → 必须后台 keep-last-N 剪枝
#   * TongBot front/left/right 必须走 three_cam T 拼图。preflight 会把
#     recipes/camera_views.py 装进 Motus，并拒绝仍会 single_view 的旧 loader。
# ============================================================
set -euo pipefail

MOTUS_ROOT="${MOTUS_ROOT:-/data1/motus/Motus}"
RECIPE_DIR="${RECIPE_DIR:-/data1/motus/recipes}"
VENV="${VENV:-/data1/motus-env}"
NPROC="${NPROC:-8}"
CONFIG_BASENAME="tongbot_00091.yaml"
CONFIG_SRC="${MOTUS_ROOT}/configs/${CONFIG_BASENAME}"
MASTER_PORT="${MASTER_PORT:-29501}"

MODE="${MODE:-smoke}"                     # check | smoke | full
BS="${BS:-4}"                             # per-card → 全局 32（8 卡）
MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_CKPTS="${KEEP_CKPTS:-2}"             # 0 = 不剪枝
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
MAX_RETRIES="${MAX_RETRIES:-20}"          # full 模式崩溃自愈次数（带 resume）
CKPT_GIB="${CKPT_GIB:-80}"                # 单份 ckpt 体积预算

OUT_ROOT="${OUT_ROOT:-/data/train_data/motus-outputs}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${OUT_ROOT}/tongbot_00091"
RUN_DIR="${RUN_ROOT}/${MODE}_${STAMP}"
LOG="${RUN_DIR}/launcher.log"
# train.py 拼出来的最终 ckpt 目录：<ckpt_dir>/<config basename>/<run_name>
CKPT_ROOT="${OUT_ROOT}/00091/tongbot_00091/tongbot_00091"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

mkdir -p "${RUN_DIR}" 2>/dev/null || die "写不了 ${RUN_ROOT}（确认 ${OUT_ROOT} 所在盘 zgyd 可写，或 OUT_ROOT=... 换盘）"

# 找最新可续跑的 ckpt（目录非空才算完整）
latest_ckpt() {
  local n
  n=$(ls -1d "${CKPT_ROOT}"/checkpoint_step_* 2>/dev/null | sed 's/.*checkpoint_step_//' | sort -n | tail -1)
  [ -n "${n}" ] || return 0
  local p="${CKPT_ROOT}/checkpoint_step_${n}"
  [ -n "$(ls -A "${p}" 2>/dev/null)" ] && echo "${p}"
}

# 生成运行时 yaml：$1=resume(可空) $2=max_steps $3=save_interval
write_runtime_config() {
  local resume="$1" steps="$2" save="$3"
  mkdir -p "${MOTUS_ROOT}/configs_run"
  RESUME_PATH="${resume}" STEPS="${steps}" SAVE="${save}" \
  CONFIG_SRC_PATH="${CONFIG_SRC}" BS="${BS}" NPROC="${NPROC}" \
  NUM_WORKERS="${NUM_WORKERS}" LOG_INTERVAL="${LOG_INTERVAL}" \
  python - "${MOTUS_ROOT}/configs_run/${CONFIG_BASENAME}" <<'PY' | tee -a "${LOG}"
import os, sys, yaml
cfg = yaml.safe_load(open(os.environ["CONFIG_SRC_PATH"]))
res = os.environ.get("RESUME_PATH") or ""
cfg["resume"]["checkpoint_path"] = res or None
cfg["training"]["max_steps"] = int(os.environ["STEPS"])
cfg["training"]["batch_size"] = int(os.environ["BS"])
cfg["system"]["save_interval"] = int(os.environ["SAVE"])
cfg["system"]["num_workers"] = int(os.environ["NUM_WORKERS"])
cfg["system"]["log_interval"] = int(os.environ["LOG_INTERVAL"])
out = sys.argv[1]
yaml.safe_dump(cfg, open(out, "w"), allow_unicode=True, sort_keys=False)
print("[cfg] {} <- resume={} steps={} bs={}x{} save={} workers={}".format(
    out, cfg["resume"]["checkpoint_path"], cfg["training"]["max_steps"],
    cfg["training"]["batch_size"], os.environ["NPROC"],
    cfg["system"]["save_interval"], cfg["system"]["num_workers"]))
PY
  RUNTIME_CONFIG="configs_run/${CONFIG_BASENAME}"
}

# ------------------------------------------------------------
# preflight
# ------------------------------------------------------------
preflight() {
  log "========== preflight =========="
  # shellcheck disable=SC1091
  [ -f "${VENV}/bin/activate" ] || die "缺 venv: ${VENV}（先跑 setup_env_h100.sh）"
  source "${VENV}/bin/activate"
  log "python=$(which python) $(python -c 'import sys;print(sys.version.split()[0])')"

  # 1) 卡数 + 干净度（check 模式只告警）
  local gpus used busy
  gpus=$(nvidia-smi -L | wc -l)
  [ "${gpus}" -ge "${NPROC}" ] || die "只有 ${gpus} 张卡，需要 ${NPROC}"
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  if [ "${used}" -gt 3000 ]; then
    busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -5)
    if [ "${MODE}" = "check" ]; then
      log "WARN 卡上不干净（合计 ${used}MiB）：${busy} —— check 模式继续（preflight 全是 CPU 活）"
    else
      die "卡上不干净（合计 ${used}MiB），会直接 OOM 且抢别人任务。先停掉：${busy}"
    fi
  else
    log "GPU: ${gpus} 张，当前空闲"
  fi

  # 2) 代码 / 配置 + TongBot 三路相机 loader
  [ -f "${CONFIG_SRC}" ] || die "缺 ${CONFIG_SRC}（推 recipes + configs）"
  [ -f "${MOTUS_ROOT}/train/train.py" ] || die "缺 ${MOTUS_ROOT}/train/train.py"
  [ -f "${RECIPE_DIR}/validate_motus_00091_h100.py" ] || die "缺 ${RECIPE_DIR}/validate_motus_00091_h100.py"
  if [ -f "${RECIPE_DIR}/install_tongbot_three_cam.py" ]; then
    python "${RECIPE_DIR}/install_tongbot_three_cam.py" --motus-root "${MOTUS_ROOT}" \
      --recipe-dir "${RECIPE_DIR}" | tee -a "${LOG}" \
      || die "install_tongbot_three_cam.py 失败（不装这个，训练会再掉回 front 单目）"
  fi
  grep -q "from data.lerobot.camera_views import resolve_image_source" \
    "${MOTUS_ROOT}/data/lerobot/lerobot_dataset.py" \
    || die "lerobot_dataset.py 未接 TongBot 三路相机，会只训 front"
  log "代码 + ${CONFIG_BASENAME} + three-cam loader ✓"

  # 3) 落点空间：峰值 = (KEEP+1) 份 ckpt
  local need_gib have_gib
  need_gib=$(( (KEEP_CKPTS > 0 ? KEEP_CKPTS : 3) * CKPT_GIB + CKPT_GIB + 40 ))
  mkdir -p "${OUT_ROOT}/00091" || die "建不了 ${OUT_ROOT}/00091"
  have_gib=$(df -BG --output=avail "${OUT_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')
  log "输出 ${OUT_ROOT}（盘上可用 ${have_gib}G，需约 ${need_gib}G）"
  [ -n "${have_gib}" ] || die "df 读不到 ${OUT_ROOT} 所在盘"
  [ "${have_gib}" -ge "${need_gib}" ] || die "${OUT_ROOT} 空间不足：需 ${need_gib}G，只有 ${have_gib}G（降 KEEP_CKPTS 或换 OUT_ROOT）"

  # 4) CPU-only 全链路校验（配置/权重/T5 缓存/lerobot 数据/归一化/collate）
  local vout="${RUN_DIR}/validate.out"
  ( cd "${MOTUS_ROOT}" && CUDA_VISIBLE_DEVICES="" python "${RECIPE_DIR}/validate_motus_00091_h100.py" \
      --config "configs/${CONFIG_BASENAME}" --samples 6 ) > "${vout}" 2>&1 || true
  tail -35 "${vout}" | tee -a "${LOG}"
  grep -q "ALL CHECKS PASSED" "${vout}" || die "validate_motus_00091_h100.py 未通过（见 ${vout}）"
  log "========== preflight 通过 =========="
}

# keep-last-N 剪枝：只删自己 CKPT_ROOT 下的 checkpoint_step_*
prune_loop() {
  [ "${KEEP_CKPTS}" -gt 0 ] || { log "KEEP_CKPTS=0，不剪枝"; return; }
  while :; do
    if [ -d "${CKPT_ROOT}" ]; then
      while read -r d; do
        case "${d}" in
          "${CKPT_ROOT}"/checkpoint_step_*)
            # 只删正在写之外目录：mtime 超过 3 分钟的才动，避免删半截
            if [ -n "$(find "${d}" -maxdepth 0 -mmin +3 2>/dev/null)" ]; then
              rm -rf "${d}" && log "PRUNE ${d}"
            fi ;;
        esac
      done < <(ls -1d "${CKPT_ROOT}"/checkpoint_step_* 2>/dev/null | sed 's/.*checkpoint_step_//' \
                 | sort -n | head -n -"${KEEP_CKPTS}" | sed "s|^|${CKPT_ROOT}/checkpoint_step_|")
    fi
    sleep 180
  done
}

# ------------------------------------------------------------
# 训练一次：$1=tag $2=steps $3=save_interval $4=resume(可空)
# ------------------------------------------------------------
run_train() {
  local tag="$1" steps="$2" save="$3" resume="${4:-}"
  local job_dir="${RUN_DIR}/${tag}" job_log
  mkdir -p "${job_dir}"
  job_log="${job_dir}/train.log"

  write_runtime_config "${resume}" "${steps}" "${save}"
  cd "${MOTUS_ROOT}"
  export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  log "START ${tag}: steps=${steps} bs=${BS} nproc=${NPROC} save=${save} resume=${resume:-<none>} ckpt=${CKPT_ROOT} log=${job_log}"

  set +e
  torchrun \
    --nnodes=1 --nproc_per_node="${NPROC}" --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
    train/train.py \
    --deepspeed configs/zero1.json \
    --config "${RUNTIME_CONFIG}" \
    --checkpoint_dir "${OUT_ROOT}/00091" \
    --run_name "tongbot_00091" \
    --report_to tensorboard \
    > "${job_log}" 2>&1
  local rc=$?
  set -e
  log "END ${tag} rc=${rc}"
  grep -Ei "OutOfMemory|CUDA out of memory|NCCL timeout|ChildFailedError|Traceback|Error" \
    "${job_log}" | tail -6 | tee -a "${LOG}" || true
  return "${rc}"
}

clean_ckpts() {
  local tag="$1"
  [ -d "${CKPT_ROOT}" ] || return 0
  log "清理 ${tag} 的 ckpt：$(du -sh "${CKPT_ROOT}" 2>/dev/null | cut -f1) ${CKPT_ROOT}"
  rm -rf "${CKPT_ROOT}" && log "已删除 ${tag} ckpt" || log "WARN 删不掉 ${CKPT_ROOT}"
}

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
log "Motus-00091 SFT — mode=${MODE} stamp=${STAMP} out=${OUT_ROOT}"
log "  bs/card=${BS} x ${NPROC} 卡 = 全局 $(( BS * NPROC )) | max_steps=${MAX_STEPS} save=${SAVE_INTERVAL} keep=${KEEP_CKPTS}"

# shellcheck disable=SC1091
if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate"; fi

if [ "${SKIP_PREFLIGHT}" != "1" ]; then
  preflight
fi
if [ "${MODE}" = "check" ]; then
  log "MODE=check：只做校验，结束"
  exit 0
fi

if [ "${MODE}" = "smoke" ]; then
  if run_train smoke 20 10 ""; then
    clean_ckpts smoke
    log "========== 冒烟通过（20 步 + 2 次存盘 + 清理路径验证）=========="
    exit 0
  else
    clean_ckpts smoke
    die "冒烟失败（见 ${RUN_DIR}/smoke/train.log）"
  fi
fi

# ---- full：冒烟 → 正式（崩溃自愈带 resume） ----
prune_loop &
PRUNE_PID=$!
log "剪枝已启动 PID=${PRUNE_PID}（保留最近 ${KEEP_CKPTS} 个 ckpt，目录 ${CKPT_ROOT}）"
trap 'kill "${PRUNE_PID}" 2>/dev/null || true' EXIT

if [ "${SKIP_SMOKE}" != "1" ]; then
  if run_train smoke 20 10 ""; then
    clean_ckpts smoke
    log "冒烟通过 → 开正式训练"
  else
    clean_ckpts smoke
    die "冒烟失败，不开正式（见 ${RUN_DIR}/smoke/train.log）"
  fi
fi

RESUME="$(latest_ckpt || true)"
if [ -n "${RESUME}" ]; then
  log "发现已有 ckpt，从 ${RESUME} 续跑"
else
  log "无历史 ckpt，从头开始"
fi

attempt=0
while :; do
  attempt=$(( attempt + 1 ))
  if [ "${attempt}" -gt "${MAX_RETRIES}" ]; then
    die "已重试 ${MAX_RETRIES} 次仍失败，放弃（最后一次日志见 ${RUN_DIR}/sft/train.log）"
  fi
  if run_train "sft" "${MAX_STEPS}" "${SAVE_INTERVAL}" "${RESUME}"; then
    log "正常跑完 ${MAX_STEPS} 步"
    break
  fi
  NEW="$(latest_ckpt || true)"
  if [ -n "${NEW}" ] && [ "${NEW}" != "${RESUME}" ]; then
    RESUME="${NEW}"
    log "崩溃 → 从 ${RESUME} 续跑（第 ${attempt} 次重试）"
  else
    log "崩溃且无新 ckpt → 30s 后原样重试（第 ${attempt} 次）"
    sleep 30
  fi
  while pgrep -f "train/train.py" >/dev/null; do
    log "还有残留 train.py 进程，等 60s"
    sleep 60
  done
done

log "========== Motus-00091 结束 =========="
