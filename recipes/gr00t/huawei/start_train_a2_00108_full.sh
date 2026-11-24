#!/bin/bash
# A2：GR00T N1.7 全量微调 00108，max_steps=100000（默认×10）
# 新输出在最大盘 /home/software（2.9T）。正在跑的那轮若已指向 /home/docker/gr00t/outputs，不要改它的进程。
set -euo pipefail

ROOT="${GR00T_ROOT:-/home/gr00t}"
SRC="${GR00T_SRC:-${ROOT}/src/Isaac-GR00T}"
# shellcheck disable=SC1091
source "${ROOT}/code/env_npu.sh"

DATASET="${DATASET:-/home/gr00t/datasets/00108}"
MODALITY_CONFIG="${MODALITY_CONFIG:-${ROOT}/code/00108_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/software/gr00t/outputs/finetune_00108_full}"
BASE_MODEL="${BASE_MODEL:-${ROOT}/pretrained/GR00T-N1.7-3B}"
LOG="${LOG:-${OUTPUT_DIR}/train_00108_a2_8npu_full.log}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29551}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-62400}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export GR00T_DIT_SDPA_MODE="${GR00T_DIT_SDPA_MODE:-math}"
export TRANSFORMERS_ATTENTION_IMPLEMENTATION="${TRANSFORMERS_ATTENTION_IMPLEMENTATION:-eager}"
# 全量微调在 64GB 上只差几百 MB，避免缓存按大块切导致 290MB 申请失败。
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"
# 子进程的 sys.executable 会落到系统 python，venv 不再生效。
# 缺的包用系统 site-packages 里的 gr00t_venv.pth 追加，不要把 venv 放到 PYTHONPATH 前面
# （否则会盖住系统 transformers 5）。
export PYTHONPATH="${SRC}:${ROOT}/shim:${ROOT}/code:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"
cd "${SRC}"

{
  echo "START $(date -Is) output=${OUTPUT_DIR} max_steps=100000 lr=1e-4 nproc=${NPROC}"
} >>"${LOG}"

# 宿主机 npu-smi 已挂进容器。exec torchrun 后 pid 不变，sidecar 跟着这份训练走。
HW_PY="${ROOT}/scripts/hw_train_monitor.py"
if [ -f "${HW_PY}" ]; then
  setsid python3 "${HW_PY}" \
    --out "${LOG}.hw.jsonl" \
    --interval 30 \
    --pid $$ \
    --devices "${ASCEND_RT_VISIBLE_DEVICES}" \
    --wait-start 180 \
    </dev/null >/dev/null 2>>"${LOG}.hw.err" &
  echo "[hw-monitor] sidecar=$! watch_pid=$$ out=${LOG}.hw.jsonl" >>"${LOG}"
fi

echo "============================================================"
echo "  GR00T N1.7 full FT — 00108 on A2 (${NPROC} NPU)"
echo "  model:     ${BASE_MODEL}"
echo "  backbone:  ${GR00T_BACKBONE_PATH}"
echo "  dataset:   ${DATASET}"
echo "  modality:  ${MODALITY_CONFIG}"
echo "  output:    ${OUTPUT_DIR}"
echo "  max_steps: 100000  lr: 1e-4  batch: ${NPROC}×1 ×accum2"
echo "  tune:      llm+visual+projector+dit  deepspeed ZeRO-2"
echo "============================================================"
export GR00T_USE_DDP="${GR00T_USE_DDP:-0}"
export GR00T_DEEPSPEED_STAGE="${GR00T_DEEPSPEED_STAGE:-2}"

test -s "${DATASET}/meta/info.json"
test -s "${DATASET}/meta/modality.json"
test -s "${MODALITY_CONFIG}"
test -s "${BASE_MODEL}/config.json"
test -x "${ROOT}/venv/bin/python"

exec "${ROOT}/venv/bin/python" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  "${ROOT}/code/launch_finetune_ascend.py" \
  --base-model-path "${BASE_MODEL}" \
  --dataset-path "${DATASET}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "${MODALITY_CONFIG}" \
  --num-gpus "${NPROC}" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name finetune_00108_full \
  --max-steps 100000 \
  --global-batch-size "${NPROC}" \
  --dataloader-num-workers 2 \
  --learning-rate 1e-4 \
  --save-steps 2000 \
  --save-total-limit 5 \
  --gradient-accumulation-steps 2 \
  --state-dropout-prob 0.2 \
  --tune-llm \
  --tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --gradient-checkpointing \
  >>"${LOG}" 2>&1
