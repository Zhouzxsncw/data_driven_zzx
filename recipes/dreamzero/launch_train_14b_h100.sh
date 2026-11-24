#!/usr/bin/env bash
# ============================================================
# h100-103 上运行：DreamZero-14B (Wan2.1-I2V-14B) TongBot 00073 SFT
# 8x H100 80GB，FSDP2 full_shard + bf16 mixed precision
#
# 用法（在 H100 上）:
#   MODE=check bash /data1/dreamzero-config/launch_train_14b_h100.sh     # 只跑 preflight（不占卡）
#   bash /data1/dreamzero-config/launch_train_14b_h100.sh                # preflight + 4 步冒烟
#   MODE=full bash /data1/dreamzero-config/launch_train_14b_h100.sh      # 冒烟通过后跑正式训练（nohup + keep-last-N）
#   GPUS=2-7 MODE=full ... bash ...                                      # 只用卡 2-7（腾出 0/1），GBS/NPROC 自动=6
#
# 可覆盖: OUT_ROOT / MBS / GBS / MAX_STEPS / SAVE_INTERVAL / KEEP_CKPTS / SKIP_PREFLIGHT / RESUME_DIR / GPUS / NPROC
# 机制与已验证的 5B 启动器一致（/data1/dreamzero-config/train_dreamzero_h100.sh
# 与 dreamzero_staging/loop_dreamzero_saturate.sh）：单进程 python，
# RLinf 自己按 cluster.component_placement.actor 起 worker（all=全部卡，GPUS 覆盖成范围）。
# ============================================================
set -euo pipefail

BASE_DIR="${BASE_DIR:-/data1}"
VENV_DIR="${BASE_DIR}/rlinf-env"
RLINF_DIR="${BASE_DIR}/RLinf"
DREAMZERO_DIR="${BASE_DIR}/dreamzero"
PRETRAINED="${BASE_DIR}/dreamzero-pretrained"
CONFIG_DIR="${BASE_DIR}/dreamzero-config"
CONFIG_NAME="tong00073_sft_dreamzero_14b"
DATASET="${DATASET:-${BASE_DIR}/dataset/00073_20260818_RM_01_yj_21}"
METADATA="${CONFIG_DIR}/metadata_00073.json"

MODE="${MODE:-smoke}"                     # check | smoke | full
# 训练用哪些物理卡（RLinf component_placement 的范围格式，如 "2-7" = 卡 2..7）。
# 空 = all（全部卡）。注意不能用 CUDA_VISIBLE_DEVICES 缩卡：RLinf 的 Cluster/Ray
# 探测无视 CVD（按物理 8 卡算 world_size），actor=all + 6 卡 GBS 会在
# validate_cfg 的 "gbs % (mbs*world_size)" 断言直接炸（实测 2026-09-11）。
GPUS="${GPUS:-}"
if [[ "${GPUS}" == *-* ]]; then
  NPROC_DEFAULT=$(( ${GPUS#*-} - ${GPUS%-*} + 1 ))
elif [ -n "${GPUS}" ]; then
  NPROC_DEFAULT=1
else
  NPROC_DEFAULT=8
fi
NPROC="${NPROC:-${NPROC_DEFAULT}}"
export NPROC                              # preflight 导入链断言按它检查可见卡数
# 14B 全量 ckpt ≈150–170G，/data1 只剩 ~370G → 默认 落到 /data（实测 1.3T 空闲）。
# /data 根目录是 root:root 755，zgyd 建不了目录；机上唯一可写的大盘目录是 /data/train_data（777）。
OUT_ROOT="${OUT_ROOT:-/data/train_data/dreamzero-outputs}"
MBS="${MBS:-1}"                           # 14B 从 micro=1 起步，OOM 就先开 enable_offload 再说
GBS="${GBS:-$(( MBS * NPROC ))}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_CKPTS="${KEEP_CKPTS:-2}"             # 0 = 不剪枝
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
# 续训：指向 checkpoints/global_step_N（目录名里的 N 会被 SFTRunner 当成起始 step）
RESUME_DIR="${RESUME_DIR:-}"

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${OUT_ROOT}/${CONFIG_NAME}"
RUN_DIR="${RUN_ROOT}/${MODE}_${STAMP}"
LOG="${RUN_DIR}/train.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

mkdir -p "${RUN_DIR}" 2>/dev/null || die "写不了 ${RUN_ROOT}（确认 ${OUT_ROOT} 已挂载且 zgyd 可写，或 OUT_ROOT=... 换盘）"

# ------------------------------------------------------------
# preflight
# ------------------------------------------------------------
preflight() {
  log "========== preflight =========="

  [ -f "${VENV_DIR}/bin/activate" ] || die "缺 venv: ${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  log "python=$(which python) $(python -c 'import sys;print(sys.version.split()[0])')"

  # 1) 卡数 + 空闲（MODE=check 只看占用，不当致命错）
  # GPUS 指定训练用哪些物理卡（如 "2-7" 腾出 0/1）时，占用检查只看这几张——
  # 范围外的卡留给别的任务，不判"不干净"。
  local gpus used busy smi_ids
  if [ -n "${GPUS}" ]; then
    if [[ "${GPUS}" == *-* ]]; then
      smi_ids=$(seq -s, "${GPUS%-*}" "${GPUS#*-}")
    else
      smi_ids="${GPUS}"
    fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${smi_ids}" | paste -sd+ | bc) \
      || die "nvidia-smi 查 GPUS=${GPUS} 失败（卡号超范围？机器卡数 $(nvidia-smi -L | wc -l)）"
    gpus=$(echo "${smi_ids}" | tr "," "\n" | wc -l)
    busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "${smi_ids}" 2>/dev/null | head -5)
    log "placement: actor=${GPUS}（${gpus} 张卡），范围外卡不检查"
  else
    gpus=$(nvidia-smi -L | wc -l)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
    busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -5)
  fi
  [ "${gpus}" -ge "${NPROC}" ] || die "只有 ${gpus} 张可用卡（GPUS=${GPUS:-全部}），需要 ${NPROC}"
  if [ "${used}" -gt 2000 ]; then
    if [ "${MODE}" = "check" ]; then
      log "WARN 卡上不干净（合计 ${used}MiB）：${busy} —— check 模式继续，真要跑得先停掉"
    else
      die "卡上不干净（合计已用 ${used}MiB）。先停掉：${busy}"
    fi
  else
    log "GPU: ${gpus} 张，当前空闲"
  fi

  # 2) 训练配置 + hydra 结构干跑
  [ -f "${RLINF_DIR}/examples/sft/config/${CONFIG_NAME}.yaml" ] \
    || die "缺 ${RLINF_DIR}/examples/sft/config/${CONFIG_NAME}.yaml（在本机执行 deploy_dreamzero_14b_h100.sh）"
  [ -f "${RLINF_DIR}/examples/sft/config/model/dreamzero_14b.yaml" ] \
    || die "缺 ${RLINF_DIR}/examples/sft/config/model/dreamzero_14b.yaml（defaults 组配置，deploy 会一并推）"
  log "config: ${CONFIG_NAME}.yaml + model/dreamzero_14b.yaml ✓"

  if [ -f "${CONFIG_DIR}/check_sft_config.py" ]; then
    # pipefail 已开：python 非 0 会让整个管道非 0 → die
    python "${CONFIG_DIR}/check_sft_config.py" \
      --config-dir "${RLINF_DIR}/examples/sft/config" \
      --config-name "${CONFIG_NAME}" | tee -a "${LOG}" \
      || die "hydra compose / 配置断言未通过"
  else
    log "WARN 缺 ${CONFIG_DIR}/check_sft_config.py，跳过 hydra 干跑"
  fi

  # 3) 14B 组件（缺哪个补哪个；ModelScope 优先，这台机 hf-mirror 的 xet 会挂死）
  local MS14="https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P/resolve/master"
  local n_shard
  n_shard=$(find "${PRETRAINED}/Wan2.1-I2V-14B-480P" -maxdepth 1 -name "diffusion_pytorch_model-*of-00007.safetensors" 2>/dev/null | wc -l)
  [ "${n_shard}" -eq 7 ] \
    || die "Wan2.1-I2V-14B-480P 分片只有 ${n_shard}/7 —— 先 nohup bash ${CONFIG_DIR}/download_14b_h100.sh &"
  [ -f "${PRETRAINED}/Wan2.1-I2V-14B-480P/config.json" ] || die "缺 Wan2.1-I2V-14B-480P/config.json"
  log "14B DiT 分片 ${n_shard}/7 + config.json ✓"
  _ensure_single "${PRETRAINED}/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" 400000000 \
    "${MS14}/Wan2.1_VAE.pth" \
    "https://hf-mirror.com/Wan-AI/Wan2.1-I2V-14B-480P/resolve/main/Wan2.1_VAE.pth"
  _ensure_single "${PRETRAINED}/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" 10000000000 \
    "https://www.modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B/resolve/master/models_t5_umt5-xxl-enc-bf16.pth" \
    "https://hf-mirror.com/Wan-AI/Wan2.2-TI2V-5B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth"
  _ensure_single "${PRETRAINED}/Wan2.1-CLIP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" 4000000000 \
    "${MS14}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    "https://hf-mirror.com/Wan-AI/Wan2.1-I2V-14B-480P/resolve/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  [ -f "${PRETRAINED}/umt5-xxl/tokenizer_config.json" ] || die "缺 umt5-xxl tokenizer: ${PRETRAINED}/umt5-xxl"
  log "T5 / CLIP / tokenizer ✓"

  # 4) 数据
  [ -d "${DATASET}/meta" ] || die "缺数据集 ${DATASET}"
  [ -f "${METADATA}" ] || die "缺 metadata: ${METADATA}"
  log "数据集 $(basename "${DATASET}") + metadata ✓"

  # 5) 导入链（14B 用 Wan2.1 的 WanVideoVAE，不是 5B 的 VAE38 —— 这里一起验）
  python - <<'PY' | tee -a "${LOG}" || die "导入链失败（见上面 traceback）"
import os
import torch
print(f"  torch {torch.__version__} CUDA={torch.cuda.is_available()} GPUs={torch.cuda.device_count()}")
assert torch.cuda.is_available() and torch.cuda.device_count() >= int(os.environ.get("NPROC", "8")), \
    f"可见 GPU {torch.cuda.device_count()} 张 < NPROC={os.environ.get('NPROC', '8')}（查 GPUS/CUDA_VISIBLE_DEVICES）"
from groot.vla.data.transform import ComposedModalityTransform
from rlinf.models.embodiment.dreamzero import get_model
from rlinf.data.datasets.dreamzero.data_transforms.tong_dual_arm import TongDualArmDataTransform
from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE, WanVideoVAE38
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel
print("  imports OK:", TongDualArmDataTransform.TAG, "| WanVideoVAE(14B) + CausalWanModel 可用")
PY

  # 6) 落点空间：实测单个 14B ckpt（dcp 分片 8 × ~28G）= 209G，不是估的 170G；
  #    剪枝是在写完新 ckpt 之后才删旧的，所以峰值 = (KEEP+1) 个 ckpt。
  local need_gib have_gib
  need_gib=$(( (KEEP_CKPTS > 0 ? KEEP_CKPTS : 3) * 210 + 210 + 50 ))
  mkdir -p "${RUN_ROOT}"
  have_gib=$(df -BG --output=avail "$(dirname "${OUT_ROOT}")" 2>/dev/null | tail -1 | tr -dc '0-9')
  log "输出 ${RUN_ROOT}（盘上可用 ${have_gib}G，需约 ${need_gib}G）"
  [ -n "${have_gib}" ] || die "df 读不到 ${OUT_ROOT} 所在盘"
  [ "${have_gib}" -ge "${need_gib}" ] || die "${OUT_ROOT} 空间不足：需 ${need_gib}G，只有 ${have_gib}G（改 OUT_ROOT 或先清 cosmos-outputs 的 optim）"

  if [ -n "${RESUME_DIR}" ]; then
    [ -d "${RESUME_DIR}/actor" ] || die "RESUME_DIR 缺 actor/: ${RESUME_DIR}"
    log "resume_dir=${RESUME_DIR}（step=$(basename "${RESUME_DIR}" | sed 's/.*global_step_//')）"
  fi

  log "========== preflight 通过 =========="
}

# 单文件兜底下载：$1=路径 $2=最小字节 $3..=候选 URL
_ensure_single() {
  local path="$1" min_bytes="$2"; shift 2
  if [ -f "${path}" ] && [ "$(stat -c %s "${path}")" -ge "${min_bytes}" ]; then
    log "$(basename "${path}") ✓ $(( $(stat -c %s "${path}") / 1024 / 1024 ))M"
    return 0
  fi
  log "$(basename "${path}") 缺失/不完整，补下载…"
  mkdir -p "$(dirname "${path}")"
  local url
  for url in "$@"; do
    if curl -fL -C - --retry 3 --retry-delay 5 --connect-timeout 20 -o "${path}.part" "${url}" \
       && [ "$(stat -c %s "${path}.part" 2>/dev/null || echo 0)" -ge "${min_bytes}" ]; then
      mv "${path}.part" "${path}"
      log "$(basename "${path}") 下载完成"
      return 0
    fi
    log "  源失败，换下一个：${url}"
    rm -f "${path}.part"
  done
  die "下载失败：${path}（所有源都试过）"
}

# ------------------------------------------------------------
# 训练（$1=tag $2=max_steps $3=mbs $4=gbs $5=save_interval）
# ------------------------------------------------------------
run_train() {
  local tag="$1" steps="$2" mbs="$3" gbs="$4" save="$5"
  local job_dir="${RUN_DIR}/${tag}"
  local job_log="${job_dir}/train.log"
  mkdir -p "${job_dir}"

  cd "${RLINF_DIR}/examples/sft"
  local extra=()
  if [ -n "${RESUME_DIR}" ]; then
    extra+=("runner.resume_dir=${RESUME_DIR}")
  fi
  if [ -n "${GPUS}" ]; then
    extra+=("cluster.component_placement.actor=${GPUS}")
  fi
  # EXTRA_OVERRIDES：空格分隔的 hydra override 直接透传（如
  # "actor.fsdp_config.gradient_checkpointing=True"——6 卡 OOM 时开 ckpt 省激活）
  if [ -n "${EXTRA_OVERRIDES:-}" ]; then
    # shellcheck disable=SC2206
    extra+=(${EXTRA_OVERRIDES})
  fi
  log "START ${tag}: steps=${steps} mbs=${mbs} gbs=${gbs} gpus=${GPUS:-all} save=${save} resume=${RESUME_DIR:-null} log=${job_log}"
  set +e
  python train_vla_sft.py \
    --config-path config/ \
    --config-name "${CONFIG_NAME}" \
    "runner.logger.log_path=${job_dir}" \
    "runner.logger.experiment_name=${CONFIG_NAME}_${tag}" \
    "runner.max_steps=${steps}" \
    "runner.log_interval=${LOG_INTERVAL}" \
    "runner.save_interval=${save}" \
    "actor.micro_batch_size=${mbs}" \
    "actor.global_batch_size=${gbs}" \
    "${extra[@]}" \
    >> "${job_log}" 2>&1
  local rc=$?
  set -e
  log "END ${tag} rc=${rc}"
  if grep -Eqi 'OutOfMemoryError|CUDA out of memory' "${job_log}"; then
    grep -E "MiB|reserved" "${job_log}" | tail -5 | tee -a "${LOG}" || true
    return 99
  fi
  return "${rc}"
}

# keep-last-N 剪枝：只删本次 run 目录下自己的 global_step_*，其余一律不碰
prune_loop() {
  local ckpt_root="$1"
  [ "${KEEP_CKPTS}" -gt 0 ] || { log "KEEP_CKPTS=0，不剪枝"; return; }
  while :; do
    if [ -d "${ckpt_root}" ]; then
      # 保留最新的 KEEP_CKPTS 个，其余删除
      while read -r d; do
        case "${d}" in
          "${ckpt_root}"/global_step_*) rm -rf "${d}" && log "PRUNE ${d}" ;;
        esac
      done < <(ls -1d "${ckpt_root}"/global_step_* 2>/dev/null | sort -V | head -n -"${KEEP_CKPTS}")
    fi
    sleep 180
  done
}

# RLinf 的 runner 会在最后一个 step 无条件存盘（save_interval 再大也挡不住），所以
# 4 步冒烟也会写出 ~209G 的 global_step_4 —— 验完立即清掉，别把 /data 吃满。
clean_smoke_ckpt() {
  local root="${RUN_DIR}/$1/${CONFIG_NAME}_$1/checkpoints"
  if [ -d "${root}" ]; then
    log "清理 $1 的 ckpt（$(du -sh "${root}" | cut -f1)）：${root}"
    rm -rf "${root}" && log "已删除 $1 ckpt" || log "WARN 删不掉 ${root}，手工确认"
  fi
}

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
log "DreamZero-14B SFT — mode=${MODE} stamp=${STAMP}"
log "OUT_ROOT=${OUT_ROOT} MBS=${MBS} GBS=${GBS} NPROC=${NPROC} GPUS=${GPUS:-all} MAX_STEPS=${MAX_STEPS} SAVE=${SAVE_INTERVAL} KEEP=${KEEP_CKPTS} RESUME=${RESUME_DIR:-null}"

export DREAMZERO_PATH="${DREAMZERO_DIR}"
export PYTHONPATH="${RLINF_DIR}:${DREAMZERO_DIR}:${PYTHONPATH:-}"
export EMBODIED_PATH="${RLINF_DIR}/examples/sft"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# 缩卡（GPUS 非空）时自动关闭 wan_video_dit DiT 算子的 torch.compile：PyTorch 2.9 下
# inductor CUDA graphs + FSDP2 梯度检查点显存异常膨胀（6 卡实测 79G/卡贴满仍 OOM，
# no-reentrant gc 还会触发 cudagraph_trees bug 崩溃）。8 卡全量训练不受影响（默认空=compile 开）。
if [ -n "${GPUS}" ] && [ "${DISABLE_TORCH_COMPILE:-}" != "true" ]; then
  DISABLE_TORCH_COMPILE=true
  log "缩卡模式 GPUS=${GPUS}：自动 DISABLE_TORCH_COMPILE=true（关 DiT compile，防 CUDA graphs 显存膨胀）"
fi
export DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-}"
export NCCL_DEBUG=WARN
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

if [ "${SKIP_PREFLIGHT}" != "1" ]; then
  preflight
else
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  log "SKIP_PREFLIGHT=1，跳过检查"
fi

if [ "${MODE}" = "check" ]; then
  log "MODE=check：preflight 已过，不占卡。可以开跑： bash $0（冒烟）/ MODE=full bash $0（正式）"
  exit 0
fi

if [ "${MODE}" = "smoke" ]; then
  # 4 步、不存 ckpt：只为验通路 + 读每卡峰值显存
  rc=0
  run_train smoke 4 "${MBS}" "${GBS}" 100000 || rc=$?
  [ "${rc}" = "0" ] || die "冒烟失败 rc=${rc}（99=OOM，看 ${RUN_DIR}/smoke/train.log）"
  log "冒烟通过。取每卡显存峰值："
  grep -Ei "memory_allocated|max_memory_allocated|Memory-|GiB" "${RUN_DIR}/smoke/train.log" | tail -10 | tee -a "${LOG}" || true
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "${LOG}"
  clean_smoke_ckpt smoke
  log "正式开训： MODE=full bash $0"
  exit 0
fi

# full：先冒烟，再把自己 nohup 成后台长跑（__inner_full 分支）
if [ "${1:-}" = "__inner_full" ]; then
  prune_loop "${RUN_DIR}/sft/${CONFIG_NAME}_sft/checkpoints" &
  PRUNE_PID=$!
  log "剪枝已启动 PID=${PRUNE_PID}（保留最近 ${KEEP_CKPTS} 个 ckpt）"
  rc=0
  run_train sft "${MAX_STEPS}" "${MBS}" "${GBS}" "${SAVE_INTERVAL}" || rc=$?
  kill "${PRUNE_PID}" 2>/dev/null || true
  exit "${rc}"
fi

rc=0
run_train smoke 4 "${MBS}" "${GBS}" 100000 || rc=$?
[ "${rc}" = "0" ] || die "冒烟失败 rc=${rc}，不进正式训练"
clean_smoke_ckpt smoke
log "冒烟通过，后台启动正式训练 ${MAX_STEPS} 步"

MODE=full STAMP="${STAMP}" nohup bash "$0" __inner_full > "${RUN_DIR}/nohup.log" 2>&1 &
MAIN_PID=$!
log "正式训练 PID=${MAIN_PID}"
log "跟进: tail -f ${RUN_DIR}/sft/train.log"
log "停:  kill ${MAIN_PID}  ；剪枝目录 ${RUN_DIR}/sft/${CONFIG_NAME}_sft/checkpoints"
