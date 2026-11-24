#!/usr/bin/env bash
# FastWAM TongBot 00036 断点续训（真 resume：optimizer / LR schedule / dataloader 位置全还原）。
#
# 与 launch_train.sh 的三点差别：
#   1) 显式传 resume=<accelerate state 目录>，trainer 走 accelerator.load_state + trainer_state.json，
#      global_step / epoch / batch_in_epoch 一并恢复（launch_train.sh 只能从头训）。
#   2) output_dir 落在 /data：一份 state ≈ 80G，根分区只剩 ~100G，写不下。
#   3) 起跑前做完整预检（state 完整性 / 显存空闲 / 磁盘余量 / 内存 / 重复进程），任一条不满足直接退出。
#
# 停跑现场：09-07 08:07 SIGTERM，step 11550/20000，loss 0.6051（action 0.0506 / video 0.5545）；
#          最新完整 state = step_010000（global_step=10000, epoch=3, batch_in_epoch=2164）。
# 用法：
#   bash start_train_00036_resume.sh              # 预检通过后后台起跑
#   DRY_RUN=1 bash start_train_00036_resume.sh    # 只跑预检，不起跑
#   DETACH=0  bash start_train_00036_resume.sh    # 前台跑（调试用）
set -euo pipefail

FASTWAM_ROOT="${FASTWAM_ROOT:-/home/xk/VLA/FastWAM}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 预检里会 cd 到 FASTWAM_ROOT，先固化脚本目录
TASK_CFG="${TASK_CFG:-tongbot_00036_3cam384_1e-4}"
CKPT_ROOT="${CKPT_ROOT:-/data/FastWAM_runs/tongbot_00036}"
RESUME="${RESUME:-${CKPT_ROOT}/checkpoints/state/step_010000}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_ROOT}}"
NPROC="${NPROC:-1}"
ACCEL_CFG="${ACCEL_CFG:-scripts/accelerate_configs/accelerate_zero1_cpuoffload_lowmem.yaml}"
# ^ 为什么默认换 lowmem（pin_memory:false）：ds_zero1_cpuoffload.json 的 offload_optimizer.pin_memory=true
#   会把 ZeRO-1 的 72G fp32 状态（master + exp_avg + exp_avg_sq）用 cudaHostAlloc 钉死，
#   在 kernel OOM 里表现为 shmem-rss ~90G 且**不可换页**。resume 时 torch.load 还要把同一份
#   72G optim_states.pt 反序列化成新的 anon 张量再拷进钉死缓冲，峰值 = 90(钉) + 72(临时) + 13(model)
#   > 188G 物理内存。2026-09-14 20:34 那次 resume 就是在这里被 global OOM 打死（exitcode -9，
#   shmem-rss:90505536kB / anon-rss:52602296kB）。没有 root 加不了 swap，改 pin_memory:false 后
#   这 90G 变成可换页 anon，峰值仍在但能被 swap 吸收。计算语义/存盘格式完全不变，只影响 H2D/D2H 带宽。
#   要回到原配置：ACCEL_CFG=scripts/accelerate_configs/accelerate_zero1_cpuoffload.yaml bash 本脚本
MIN_FREE_GIB="${MIN_FREE_GIB:-95}"          # 一份 state 80G + 一份 weights 12G，另留 3G 余量
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-150}"       # 峰值 ≈ 72(临时 ckpt) + 72(offload) + 13(model) + 余量
# 内存账（2026-09-14 实测，机器 188G RAM + 64G swap，空闲基线 machine_anon 仅 4G）：
#   resume 后父进程 RssAnon ≈105-127G（72G ZeRO-1 CPU offload 优化器状态 + 模型/框架 + 载入瞬时副本）。
#   默认 fork 出的每个 dataloader worker 初始与父进程 100% 共享页（RSS≈107G/个，纯虚数），但 worker
#   写脏继承页即触发 COW 复制：mem_probe 记到 machine_anon 在 worker 起来后 40s 内 117→150→180G，
#   于是 4 worker 和 2 worker 都在同一位置被 global OOM 打死（降 worker 数只是推迟越界）。
#   对策是把 dataloader 的多进程上下文换成 spawn：子进程不继承父进程地址空间，只收 1.3MiB 数据集
#   pickle（RobotVideoDataset 实测可 pickle、构建 7.8s），代价是每个 worker 多 1-2s 启动。
#   worker 数因此可以维持原配方的 4（吞吐/样本顺序都不变：顺序由主进程的 ResumableEpochSampler(seed)
#   决定；数据集里 np.random 只用于调试取样与异常兜底）。
NUM_WORKERS="${NUM_WORKERS:-4}"
export FASTWAM_DLOADER_MP="${FASTWAM_DLOADER_MP:-spawn}"   # fork=原行为（需自备内存余量）
MIN_SWAPFREE_GIB="${MIN_SWAPFREE_GIB:-30}"  # 机器无 swap 时把本项设 0 跳过
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-5000}"
DETACH="${DETACH:-1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

log() { printf '%s %s\n' "$1" "$2"; }
die() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
ok()  { echo "  ✓ $*"; }
warn(){ echo "  ! $*"; }

# ---------- 环境（与 launch_train.sh 保持一致） ----------
[[ -f "${FASTWAM_ROOT}/env.sh" ]] || die "missing ${FASTWAM_ROOT}/env.sh（先跑 setup_env.sh）"
# shellcheck disable=SC1091
source "${FASTWAM_ROOT}/env.sh" >/dev/null
if [[ -x "${HOME}/miniconda3/envs/fastwam/bin/python" ]]; then
  PYTHON="${HOME}/miniconda3/envs/fastwam/bin/python"
elif [[ -x "${HOME}/anaconda3/envs/fastwam/bin/python" ]]; then
  PYTHON="${HOME}/anaconda3/envs/fastwam/bin/python"
else
  die "no fastwam conda env python"
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  CUDA_HOME=""
  for _cuda_home in \
    "${HOME}/miniconda3/envs/lingbotva" \
    "${HOME}/miniconda3/envs/lingbotvla" \
    /usr/local/cuda; do
    if [[ -x "${_cuda_home}/bin/nvcc" ]]; then CUDA_HOME="${_cuda_home}"; break; fi
  done
fi
[[ -n "${CUDA_HOME}" ]] || die "need CUDA_HOME with nvcc for DeepSpeed"
export CUDA_HOME
# nvcc 给 DeepSpeed 用；accelerate/python 必须仍是 fastwam，不能被 lingbotva 抢 PATH
export PATH="$(dirname "${PYTHON}"):${CUDA_HOME}/bin:${PATH}"
# DeepSpeed CPU Adam 链接需要 pip 包里的 libcurand 等
_NVLIB="$(dirname "${PYTHON}")/../lib/python3.10/site-packages/nvidia"
if [[ -d "${_NVLIB}" ]]; then
  _nvpath=""
  for d in "${_NVLIB}"/*/lib; do [[ -d "$d" ]] || continue; _nvpath="${d}:${_nvpath}"; done
  export LIBRARY_PATH="${_nvpath}${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${_nvpath}${LD_LIBRARY_PATH:-}"
fi
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ROOT}/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export WANDB_MODE=offline

echo "PYTHON=${PYTHON}  CUDA_HOME=${CUDA_HOME}"

# ---------- 预检 ----------
echo "[preflight]"
cd "${FASTWAM_ROOT}"
[[ -f "${ACCEL_CFG}" ]] || die "missing ${ACCEL_CFG}"
ok "accelerate config: ${ACCEL_CFG}"
"${PYTHON}" -c "import fastwam, hydra, omegaconf" 2>/dev/null \
  || die "fastwam 包不可导入（pip install -e . 掉了？）"
ok "fastwam import"
[[ -f "${RESUME}/trainer_state.json" ]] || die "resume 目录缺 trainer_state.json：${RESUME}"
[[ -s "${RESUME}/pytorch_model/mp_rank_00_model_states.pt" ]] || die "resume 目录缺 mp_rank_00_model_states.pt"
[[ $(( $(stat -c %s "${RESUME}/pytorch_model/mp_rank_00_model_states.pt") )) -gt 1000000000 ]] \
  || die "mp_rank_00_model_states.pt 过小，state 疑似截断"
ls "${RESUME}"/pytorch_model/*optim_states.pt >/dev/null 2>&1 || die "resume 目录缺 optimizer states（不是真·可续训 state）"
[[ -f "${RESUME}/scheduler.bin" ]] || warn "缺 scheduler.bin：LR schedule 会从头重跑"
[[ -f "${RESUME}/random_states_0.pkl" ]] || warn "缺 random_states_0.pkl：数据流/RNG 不连续"
RESUME_STEP="$("${PYTHON}" -c "import json;print(json.load(open('${RESUME}/trainer_state.json'))['global_step'])")"
ok "state 完整，resume from global_step=${RESUME_STEP}"
if [[ "$(cd "${RESUME}" && pwd -P)" == "$(cd "${OUTPUT_DIR}/checkpoints/state/step_010000" 2>/dev/null && pwd -P || echo NONE)" ]]; then
  echo "  (新存盘与 resume 源在同一棵树：${OUTPUT_DIR}/checkpoints/{weights,state})"
fi
[[ -d "${FASTWAM_ROOT}/data/text_embeds_cache/tongbot_00036" ]] \
  || die "缺 T5 文本嵌入缓存 data/text_embeds_cache/tongbot_00036（先跑 launch_train.sh 的 text 预计算）"
ok "text embed cache"
[[ -f "${FASTWAM_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" ]] \
  || die "缺 ActionDiT 骨干 ckpt"
ok "ActionDiT backbone"
DS_DIR="$(grep -A2 'dataset_dirs:' "${FASTWAM_ROOT}/configs/task/${TASK_CFG}.yaml" \
  | sed -n 's/^[[:space:]]*-[[:space:]]*\([^[:space:]]*\).*/\1/p' | head -1)"
[[ -n "${DS_DIR}" && -d "${DS_DIR}" ]] || die "数据集目录不存在：${DS_DIR:-<未解析>}"
ok "dataset ${DS_DIR}"

if command -v nvidia-smi >/dev/null; then
  gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if (( gpu_used > MAX_GPU_USED_MIB )); then
    if [[ "${FORCE}" == "1" ]]; then warn "显存已用 ${gpu_used} MiB（FORCE=1 继续）"
    else die "显存已用 ${gpu_used} MiB > ${MAX_GPU_USED_MIB} MiB；先停掉占用服务或 FORCE=1"; fi
  else ok "GPU 空闲（used=${gpu_used} MiB）"; fi
fi
DF_TARGET="${OUTPUT_DIR}"; while [[ ! -d "${DF_TARGET}" && "${DF_TARGET}" != "/" ]]; do DF_TARGET="$(dirname "${DF_TARGET}")"; done
free_gib="$(df -BG --output=avail "${DF_TARGET}" | tail -1 | tr -dc '0-9')"
[[ -n "${free_gib}" ]] || die "无法读取 ${OUTPUT_DIR} 磁盘余量"
if (( free_gib < MIN_FREE_GIB )); then
  if [[ "${FORCE}" == "1" ]]; then warn "${OUTPUT_DIR} 仅剩 ${free_gib}G < ${MIN_FREE_GIB}G（FORCE=1 继续）"
  else die "磁盘仅剩 ${free_gib}G < ${MIN_FREE_GIB}G：一份 state≈80G，先清理或接 keep_last 剪枝"; fi
else ok "磁盘余量 ${free_gib}G"
fi
avail_gib="$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)"
if (( avail_gib < MIN_AVAIL_GIB )); then
  if [[ "${FORCE}" == "1" ]]; then warn "可用内存 ${avail_gib}G < ${MIN_AVAIL_GIB}G（FORCE=1 继续）"
  else die "可用内存 ${avail_gib}G < ${MIN_AVAIL_GIB}G：载入 72G optimizer state 可能被打断"; fi
else ok "可用内存 ${avail_gib}G"
fi
# pin_memory:false 之后，offload 状态变成可换页 anon，swap 是真实余量的一部分（上次 OOM 前 64G swap 被吃光）
swapfree_gib="$(awk '/SwapFree/{printf "%d", $2/1048576}' /proc/meminfo)"
if (( swapfree_gib < MIN_SWAPFREE_GIB )); then
  if [[ "${FORCE}" == "1" ]]; then warn "SwapFree ${swapfree_gib}G < ${MIN_SWAPFREE_GIB}G（FORCE=1 继续）"
  else die "SwapFree 仅 ${swapfree_gib}G < ${MIN_SWAPFREE_GIB}G：峰值吸收不住，先腾出 swap（如关掉吃内存的桌面进程）"; fi
else ok "SwapFree ${swapfree_gib}G"
fi
# 只按本 task 判定重复：仓库里其它 recipe（gigaworld_policy 等）也用 scripts/train.py 这个名字，
# 但它们不是本次续训的冲突者；GPU 是否被占另有上面的显存闸。
if pgrep -af "scripts/train.py" 2>/dev/null | grep -q "task=${TASK_CFG}"; then
  [[ "${FORCE}" == "1" ]] || die "已有本 task 的训练进程在跑：$(pgrep -af "scripts/train.py" | grep "task=${TASK_CFG}" | head -1)"
  warn "已有本 task 训练进程，FORCE=1 放行"
fi
other="$(pgrep -af "scripts/train.py" 2>/dev/null | grep -v "task=${TASK_CFG}" | head -1 || true)"
[[ -n "${other}" ]] && warn "机器上还有别的 train.py 进程（不阻塞，但会抢内存）：${other:0:120}"
ok "无重复的本 task 训练进程"
# resume 后 LR 会因 param_groups 被 load_state_dict 换对象而冻结，必须带 lr-rebind 补丁
if [[ "${REQUIRE_LR_PATCH:-1}" == "1" ]]; then
  "${PYTHON}" "${SELF_DIR}/patch_resume_lr_rebind.py" --check >/dev/null 2>&1 \
    || die "trainer.py 未打 LR-rebind 补丁（续训后半程 cosine 会冻结）：先跑 python ${SELF_DIR}/patch_resume_lr_rebind.py"
  ok "LR-rebind 补丁已就位"
fi
# dataloader 必须能用 spawn 起 worker，否则 resume 后 fork 的 COW 脏页会把 188G+64G swap 吃光
if [[ "${REQUIRE_MP_PATCH:-1}" == "1" ]]; then
  if [[ "${FASTWAM_DLOADER_MP}" == "spawn" ]]; then
    "${PYTHON}" "${SELF_DIR}/patch_dataloader_mp_ctx.py" --check >/dev/null 2>&1 \
      || die "trainer.py 未打 dataloader-mp-ctx 补丁：spawn 无法生效，resume 会被 COW 脏页 OOM 打死；先跑 python ${SELF_DIR}/patch_dataloader_mp_ctx.py"
  fi
  ok "dataloader-mp-ctx 补丁就位（FASTWAM_DLOADER_MP=${FASTWAM_DLOADER_MP}）"
fi

if [[ "${DRY_RUN}" == "1" ]]; then echo "[dry-run] 预检通过，未起跑"; exit 0; fi

# ---------- 起跑 ----------
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}/checkpoints/weights" "${OUTPUT_DIR}/checkpoints/state"
LOG="${LOG_DIR}/train_${RUN_ID}.log"
echo "${RESUME}" > "${LOG_DIR}/resume_from_${RUN_ID}.txt"
echo "[train] task=${TASK_CFG} resume_step=${RESUME_STEP} nproc=${NPROC} num_workers=${NUM_WORKERS} run_id=${RUN_ID}"
echo "[train] accel_cfg=${ACCEL_CFG} dloader_mp=${FASTWAM_DLOADER_MP} diag=${FASTWAM_RESUME_DIAG:-0}"
echo "[train] output_dir=${OUTPUT_DIR}"
echo "[train] log=${LOG}"

CMD=(env RUN_ID="${RUN_ID}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "FASTWAM_RESUME_DIAG=${FASTWAM_RESUME_DIAG:-0}"
  "FASTWAM_DLOADER_MP=${FASTWAM_DLOADER_MP}"
  "$(dirname "${PYTHON}")/accelerate" launch
  --config_file "${ACCEL_CFG}"
  --num_processes "${NPROC}"
  scripts/train.py
  "output_dir=${OUTPUT_DIR}"
  "wandb.name=${TASK_CFG}"
  "task=${TASK_CFG}"
  "model.redirect_common_files=false"
  "num_workers=${NUM_WORKERS}"
  "resume=${RESUME}")
if [[ "${DETACH}" == "1" ]]; then
  setsid nohup "${CMD[@]}" >>"${LOG}" 2>&1 &
  pid=$!
  echo "$pid" > "${LOG_DIR}/train_${RUN_ID}.pid"
  ln -sfn "${LOG}" "${LOG_DIR}/latest.log"
  echo "[train] pid=${pid} detached  (tail -f ${LOG})"
  if [ -f /home/xk/data_Driven/scripts/start_hw_monitor.sh ]; then
    # shellcheck disable=SC1091
    . /home/xk/data_Driven/scripts/start_hw_monitor.sh
    start_hw_monitor "${LOG}" "${CUDA_VISIBLE_DEVICES:-}" "${pid}" || true
  fi
else
  if [ -f /home/xk/data_Driven/scripts/start_hw_monitor.sh ]; then
    # shellcheck disable=SC1091
    . /home/xk/data_Driven/scripts/start_hw_monitor.sh
    start_hw_monitor "${LOG}" "${CUDA_VISIBLE_DEVICES:-}" $$ || true
  fi
  "${CMD[@]}"
fi
