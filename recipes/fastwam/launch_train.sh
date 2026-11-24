#!/usr/bin/env bash
# FastWAM TongBot 训练入口。
#   TASK=00036 本机；TASK=00073 gpu-node1（可 WAIT_GPU=1 等现有作业释放显存）
set -euo pipefail

TASK="${TASK:?set TASK=00036 or 00073}"
FASTWAM_ROOT="${FASTWAM_ROOT:-/home/xk/VLA/FastWAM}"
NPROC="${NPROC:-1}"
WAIT_GPU="${WAIT_GPU:-0}"
WAIT_BELOW_MIB="${WAIT_BELOW_MIB:-8000}"
STOP_COSMOS3="${STOP_COSMOS3:-0}"

if [[ ! -f "${FASTWAM_ROOT}/env.sh" ]]; then
  echo "missing ${FASTWAM_ROOT}/env.sh ，先跑 setup_env.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${FASTWAM_ROOT}/env.sh"
export PATH="${HOME}/miniconda3/envs/fastwam/bin:${HOME}/anaconda3/envs/fastwam/bin:${PATH}"
hash -r 2>/dev/null || true
if [[ -x "${HOME}/miniconda3/envs/fastwam/bin/python" ]]; then
  PYTHON="${HOME}/miniconda3/envs/fastwam/bin/python"
elif [[ -x "${HOME}/anaconda3/envs/fastwam/bin/python" ]]; then
  PYTHON="${HOME}/anaconda3/envs/fastwam/bin/python"
else
  PYTHON="$(command -v python)"
fi
echo "PYTHON=${PYTHON}"

case "${TASK}" in
  00036) TASK_CFG=tongbot_00036_3cam384_1e-4 ;;
  00073) TASK_CFG=tongbot_00073_3cam384_1e-4 ;;
  *) echo "TASK must be 00036 or 00073" >&2; exit 1 ;;
esac

if [[ "${STOP_COSMOS3}" == "1" ]]; then
  echo "[gpu] stopping Cosmos3 infer server to free VRAM"
  pkill -f 'serve_tong_cosmos3.py' || true
  sleep 3
fi

if [[ "${WAIT_GPU}" == "1" ]]; then
  echo "[gpu] waiting until memory.used < ${WAIT_BELOW_MIB} MiB"
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    echo "[gpu] used=${used} MiB  $(date '+%F %T')"
    if [[ "${used}" -lt "${WAIT_BELOW_MIB}" ]]; then
      break
    fi
    sleep 60
  done
fi

cd "${FASTWAM_ROOT}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ROOT}/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export WANDB_MODE=offline
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  CUDA_HOME=""
  for _cuda_home in \
    "${HOME}/miniconda3/envs/lingbotva" \
    "${HOME}/anaconda3/envs/lingbotvla" \
    "${HOME}/miniconda3/envs/lingbotvla" \
    "${HOME}/anaconda3/envs/starVLA" \
    /usr/local/cuda; do
    if [[ -x "${_cuda_home}/bin/nvcc" ]]; then
      CUDA_HOME="${_cuda_home}"
      break
    fi
  done
fi
export CUDA_HOME="${CUDA_HOME:?need CUDA_HOME with nvcc for DeepSpeed}"
# nvcc 给 DeepSpeed 用；accelerate/python 必须仍是 fastwam，不能被 lingbotvla 抢 PATH
export PATH="$(dirname "${PYTHON}"):${CUDA_HOME}/bin:${PATH}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "accelerate=$(command -v accelerate)"
# DeepSpeed CPU Adam 链接需要 pip 包里的 libcurand
_NVLIB="$(dirname "${PYTHON}")/../lib/python3.10/site-packages/nvidia"
if [[ -d "${_NVLIB}" ]]; then
  _nvpath=""
  for d in "${_NVLIB}"/*/lib; do
    [[ -d "$d" ]] || continue
    _nvpath="${d}:${_nvpath}"
  done
  export LIBRARY_PATH="${_nvpath}${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${_nvpath}${LD_LIBRARY_PATH:-}"
fi

if [[ ! -f "${FASTWAM_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" ]]; then
  echo "[preprocess] ActionDiT backbone (CPU, reuse Wan2.2)"
  "${PYTHON}" scripts/preprocess_action_dit_backbone.py \
    --model-config configs/model/fastwam.yaml \
    --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
    --device cpu \
    --dtype bfloat16
fi

if [[ ! -d "${FASTWAM_ROOT}/data/text_embeds_cache/tongbot_${TASK}" ]]; then
  echo "[text] precompute T5 embeddings for ${TASK_CFG}"
  "${PYTHON}" scripts/precompute_text_embeds.py task="${TASK_CFG}" model.redirect_common_files=false
else
  echo "[text] cache exists, skip"
fi

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_DIR="${FASTWAM_ROOT}/runs/${TASK_CFG}/${RUN_ID}"
LOG="${LOG_DIR}/train.log"
mkdir -p "${LOG_DIR}"
echo "[train] task=${TASK_CFG} nproc=${NPROC} run_id=${RUN_ID}"
echo "[train] log=${LOG}"

DETACH="${DETACH:-1}"
ACCEL_CFG="${ACCEL_CFG:-scripts/accelerate_configs/accelerate_zero1_cpuoffload.yaml}"
CMD=(env RUN_ID="${RUN_ID}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "$(dirname "${PYTHON}")/accelerate" launch
  --config_file "${ACCEL_CFG}"
  --num_processes "${NPROC}"
  scripts/train.py
  "output_dir=./runs/${TASK_CFG}/${RUN_ID}"
  "wandb.name=${TASK_CFG}"
  "task=${TASK_CFG}"
  "model.redirect_common_files=false")
if [[ "${DETACH}" == "1" ]]; then
  nohup "${CMD[@]}" >"${LOG}" 2>&1 &
  echo "[train] pid=$! detached"
  echo "$!" >"${LOG_DIR}/train.pid"
else
  "${CMD[@]}"
fi
