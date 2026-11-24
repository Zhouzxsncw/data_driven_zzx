#!/bin/bash
# gpu-node1 上运行：等待 X-VLA 00073 训练（~50000 it）退出并释放 GPU 后，
# 自动启动 PI0-FAST TongBot 00073 训练。
# 日志: /home/cqcigai/pi0fast_00073_watcher.log + /home/cqcigai/train_pi0fast_00073.log
set -uo pipefail

REPO=/home/cqcigai/cigai-agilex-train-openpi
YAML=$REPO/configs/pi0fast/train_pi0fast_00073_1gpu.yaml
TRAIN_LOG=/home/cqcigai/train_pi0fast_00073.log
XVLA_MARKER="X-VLA"

log() { echo "$(date '+%F %T') [watcher] $*"; }

log "启动：等待 X-VLA 00073 训练退出（每 5 分钟轮询）"
while true; do
    # X-VLA 训练进程判定：XVLA conda env 的 python（进程命令行为
    # /home/cqcigai/anaconda3/envs/XVLA/bin/python -u train.py ...）
    if pgrep -f "XVLA/bin/python" >/dev/null 2>&1; then
        log "X-VLA 仍在训练，继续等待"
    else
        # 进程没了：确认显存已释放（<10GB 视为释放完成）
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "${USED:-99999}" -lt 10000 ]; then
            log "X-VLA 已退出且显存已释放（${USED}MiB），启动 PI0-FAST 训练"
            break
        fi
        log "X-VLA 进程不在但显存仍 ${USED}MiB，等待释放"
    fi
    sleep 300
done

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src"
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

log "启动 PI0-FAST 00073 训练 -> $TRAIN_LOG"
"$REPO/.venv/bin/python" scripts/train_from_yaml.py \
    --config "$YAML" \
    > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
log "训练 pid=$TRAIN_PID，watcher 退出（训练独立运行）"
