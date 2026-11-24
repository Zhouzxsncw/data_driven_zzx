#!/bin/bash
# PI0-FAST TongBot 00073 训练部署脚本（gpu-node1，从 gpu-master 执行）
# 传输内容: pi0_fast_base 权重(11GB) / FAST tokenizer HF缓存 / config补丁 / 训练YAML / watcher脚本
# 部署后 watcher 会等 X-VLA(00073) 训练退出后自动启动 PI0-FAST 训练。
set -euo pipefail

NODE=gpu-node1
REPO=/home/xk/VLA/openpi/cigai-agilex-train-openpi
REMOTE_REPO=/home/cqcigai/cigai-agilex-train-openpi
LOCAL_TRAIN_DIR=/home/xk/data_Driven/recipes/pi0fast/train

echo "==> [1/5] rsync pi0_fast_base 权重 (11GB)"
ssh "$NODE" "mkdir -p ~/.cache/openpi/openpi-assets/checkpoints"
rsync -a --info=progress2 \
    /home/xk/agx/PI0.5/pretrained_weight/openpi-assets/checkpoints/pi0_fast_base/ \
    "$NODE:~/.cache/openpi/openpi-assets/checkpoints/pi0_fast_base/"

echo "==> [2/5] rsync FAST tokenizer HF 缓存"
ssh "$NODE" "mkdir -p ~/.cache/huggingface/hub"
rsync -a \
    ~/.cache/huggingface/hub/models--physical-intelligence--fast/ \
    "$NODE:~/.cache/huggingface/hub/models--physical-intelligence--fast/"

echo "==> [3/5] rsync config 补丁脚本与训练 YAML/watcher"
ssh "$NODE" "mkdir -p $REMOTE_REPO/configs/pi0fast"
rsync -a \
    "$LOCAL_TRAIN_DIR/patch_config_fast_tongbot.py" \
    "$LOCAL_TRAIN_DIR/train_pi0fast_00073_1gpu.yaml" \
    "$LOCAL_TRAIN_DIR/watch_and_train_00073_node1.sh" \
    "$NODE:$REMOTE_REPO/configs/pi0fast/"

echo "==> [4/5] 应用 config.py 补丁 + 环境自检"
ssh "$NODE" "cd $REMOTE_REPO && python3 configs/pi0fast/patch_config_fast_tongbot.py --repo $REMOTE_REPO \
    && .venv/bin/python -c '
import importlib.util, sys
spec = importlib.util.find_spec(\"torchcodec\")
print(\"torchcodec:\", \"installed\" if spec else \"pyav fallback\")
import jax
print(\"jax:\", jax.__version__, jax.devices())
'"

echo "==> [5/5] 安装 watcher（等 X-VLA 退出后自动启动训练）"
ssh "$NODE" "cd $REMOTE_REPO && setsid nohup bash configs/pi0fast/watch_and_train_00073_node1.sh \
    > /home/cqcigai/pi0fast_00073_watcher.log 2>&1 & echo watcher_pid=\$!"
echo "部署完成。watcher 日志: gpu-node1:/home/cqcigai/pi0fast_00073_watcher.log"
