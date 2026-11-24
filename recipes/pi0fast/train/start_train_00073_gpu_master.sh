#!/bin/bash
# PI0-FAST TongBot 00073 本机热启动（从 node1 ckpt-10000 params）
# 前置:
#   1. python3 recipes/pi0fast/train/patch_config_fast_tongbot.py --repo $REPO
#   2. /home/xk/VLA/node1_ckpts/00073_pi0fast/checkpoints/10000/pretrained_model/params
#   3. 数据集 /home/xk/dataset/00073_20260818_RM_01_yj_21
# 监控: tail -f $LOG
set -euo pipefail

REPO=/home/xk/VLA/openpi/cigai-agilex-train-openpi
YAML=/home/xk/data_Driven/recipes/pi0fast/train/train_pi0fast_00073_local_from10000.yaml
LOG=/home/xk/VLA/openpi/train_pi0fast_00073_from10000.log
PARAMS=/home/xk/VLA/node1_ckpts/00073_pi0fast/checkpoints/10000/pretrained_model/params

if [ ! -d "$PARAMS" ]; then
    echo "[ERROR] 找不到 10000 params: $PARAMS" >&2
    exit 1
fi
if [ ! -d /home/xk/dataset/00073_20260818_RM_01_yj_21 ]; then
    echo "[ERROR] 找不到 00073 数据集" >&2
    exit 1
fi

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src"
export HF_HUB_OFFLINE=1
# train_state JIT 就要 ~47GB；卡上还挂 walloss/pi05 时 0.70≈68GB 预分配
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-0}" -gt 35000 ]; then
    echo "[ERROR] GPU 已用 ${USED}MiB，init_train_state 需要 ~47GB。先停 tau0wm 这类大服务再开。" >&2
    exit 1
fi
echo "GPU used ${USED}MiB, XLA fraction $XLA_PYTHON_CLIENT_MEM_FRACTION"

setsid nohup "$REPO/.venv/bin/python" scripts/train_from_yaml.py \
    --config "$YAML" \
    > "$LOG" 2>&1 &
PID=$!
echo "started pid=$PID"
echo "log: $LOG"
echo "ckpt: $REPO/checkpoints/00073_pi0fast_bs16_from10000"
# sidecar: GPU/PPU/NPU util+mem → ${LOG}.hw.jsonl（不碰训练；HW_MONITOR=0 关）
# shellcheck disable=SC1091
if [ -f /home/xk/data_Driven/scripts/start_hw_monitor.sh ]; then
  . /home/xk/data_Driven/scripts/start_hw_monitor.sh
  start_hw_monitor "$LOG" "${CUDA_VISIBLE_DEVICES:-}" "$PID" || true
fi
sleep 8
tail -20 "$LOG" || true
