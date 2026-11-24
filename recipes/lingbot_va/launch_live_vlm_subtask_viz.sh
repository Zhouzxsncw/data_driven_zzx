#!/usr/bin/env bash
# 真机 live：相机 + π0.5 PaliGemma 生成 subtask 叠字。可选 --robot 只读瑞尔曼。
# 不占动作服务端口；与 VA/π0.5 推理客户端不要同时抢相机。
set -euo pipefail
ROOT=/home/xk/VLA
PY="${ROOT}/openpi/cigai-agilex-train-openpi/.venv/bin/python"
export PYTHONPATH="${ROOT}/openpi/cigai-agilex-train-openpi:${ROOT}/openpi/cigai-agilex-train-openpi/src:${ROOT}/openpi/cigai-agilex-train-openpi/scripts:${ROOT}/infer:${PYTHONPATH:-}"
export PALIGEMMA_TOKENIZER_PATH="${ROOT}/openpi/models/pi05_fixed/tokenizer.model"
# 优先 subtask CE best；其次 pi05_fixed PG；再 000490
WEIGHT="${WEIGHT:-}"
if [[ -z "${WEIGHT}" ]]; then
  if [[ -f ${ROOT}/openpi/models/e2_subtask_ce_best/model.safetensors ]]; then
    WEIGHT=${ROOT}/openpi/models/e2_subtask_ce_best
  elif [[ -f ${ROOT}/openpi/models/pi05_fixed_paligemma/model.safetensors ]]; then
    WEIGHT=${ROOT}/openpi/models/pi05_fixed_paligemma
  else
    WEIGHT=${ROOT}/openpi/models/pi05_pg_only_000490
  fi
fi
exec "${PY}" "${ROOT}/lingbot-va-infer/live_vlm_subtask_viz.py" \
  --weight-dir "${WEIGHT}" \
  --config "${ROOT}/infer/infer/config/lingbot/lingbot_va_tongbot_16d_00036_local.yaml" \
  "$@"
