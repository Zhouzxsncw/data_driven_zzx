#!/bin/bash
# PI0-FAST TongBot 00036 训练启动脚本（gpu-master 单卡，与 motus 推理服务共卡）
# 前置:
#   1. python3 recipes/pi0fast/train/patch_config_fast_tongbot.py   （注册预设，幂等）
#   2. ~/.cache/openpi/openpi-assets/checkpoints/pi0_fast_base -> 预训练权重（已预置）
#   3. HF 缓存 ~/.cache/huggingface/hub/models--physical-intelligence--fast（已就绪）
# 监控: tail -f $LOG；checkpoint 在 $REPO/checkpoints/<exp_name>/
set -euo pipefail

REPO=/home/xk/VLA/openpi/cigai-agilex-train-openpi
YAML=/home/xk/data_Driven/recipes/pi0fast/train/train_pi0fast_00036_1gpu.yaml
LOG=/home/xk/VLA/openpi/train_pi0fast_00036.log

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src"
# FAST tokenizer 已在 HF 缓存，离线加载避免运行时网络抖动
export HF_HUB_OFFLINE=1
# 与 PI0-FAST 推理服务共卡：训练预分配 ~73GB，推理服务用剩余 ~20GB
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {exit !(s>70000)}'; then
    echo "警告: GPU 已用 >70GB，剩余显存可能不足" >&2
fi

setsid nohup "$REPO/.venv/bin/python" scripts/train_from_yaml.py \
    --config "$YAML" \
    > "$LOG" 2>&1 &
PID=$!
echo "started pid=$PID"
echo "log: $LOG"
# sidecar: GPU/PPU/NPU util+mem → ${LOG}.hw.jsonl（不碰训练；HW_MONITOR=0 关）
# shellcheck disable=SC1091
if [ -f /home/xk/data_Driven/scripts/start_hw_monitor.sh ]; then
  . /home/xk/data_Driven/scripts/start_hw_monitor.sh
  start_hw_monitor "$LOG" "${CUDA_VISIBLE_DEVICES:-}" "$PID" || true
fi
sleep 5
tail -5 "$LOG" || true
