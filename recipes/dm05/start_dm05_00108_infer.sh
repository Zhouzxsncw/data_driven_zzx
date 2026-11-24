#!/usr/bin/env bash
# gpu-master 上起 DM0.5 HTTP 推理。相机采集速率不动。
# 服务允许接收头相机历史。窗口 5 秒：1 Hz × 5 槽。当前三路仍是最新帧。
# 开不开历史由客户端 dm05_send_history 决定：不送 history_images 就不插历史 token。
#
#   bash recipes/dm05/start_dm05_00108_infer.sh
#   DM05_CKPT=/path/to/checkpoint-xxxx bash recipes/dm05/start_dm05_00108_infer.sh
set -euo pipefail
REPO="${DM05_REPO:-/home/xk/VLA/opendm}"
CKPT="${DM05_CKPT:-$REPO/user_checkpoints/dm05_sft_00108/checkpoint-2000}"
PORT="${DM05_PORT:-7891}"
cd "$REPO"
source /home/xk/miniconda3/etc/profile.d/conda.sh
conda activate opendm
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
exec bash "$REPO/script/dm05_launcher.sh" \
  --exp "$REPO/playground/dm05_sft_00108.py" \
  --nproc_per_node 1 \
  --task inference \
  --data-config.is-history \
  --inference-config.max-history-images 5 \
  --inference-config.port "$PORT" \
  --trainer-config.model-max-length 2048 \
  --model-config.model-name-or-path "$CKPT" \
  --model-config.llm-attn-implementation sdpa \
  --model-config.action-attn-implementation sdpa \
  --data-config.dataset-name tongbot_00108
