#!/usr/bin/env bash
set -euo pipefail
cd /home/xk/VLA/X-VLA

# X-VLA 00073 修复重训：action_mode=auto（AutoActionSpace，16D 纯 MSE）
# 替代原 ee6d 训练（runnings/00073_tongbot 权重右j3通道被 BCE 压扁，不可用）。
# 超参与 gpu-node1 原 00073 run 完全一致，唯一差异 = --models 指向 X-VLA-Pt-auto。
#
# 2026-09-13 batch_size 32 -> 16 并断点续训：
#   单张 RTX PRO 6000 同时扛 bs=32 训练（89.9G 显存 / 100% util / 顶 300W 功率墙）
#   和双 4K 桌面合成，桌面被饿死。减半 batch 腾显存、降单步耗时。
#   RESUME 指向上游 train.py 原生不支持，需先打 patch_train_resume.py（幂等，本脚本会自检）。
#   注意：X-VLA ckpt 只存权重，不存 optimizer/RNG —— 续训时 AdamW 矩量从 0 重启，
#   LR schedule 靠 state.json 的 global_step 恢复（已验证 cosine 连续）。

REPO=/home/xk/VLA/X-VLA
WORKSPACE=/home/xk/data_Driven
OUT=./runnings/00073_tongbot_auto
BATCH_SIZE=${BATCH_SIZE:-16}
RESUME=${RESUME-$OUT/ckpt-5000}

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/home/xk/miniconda3/envs/XVLA/bin/python
cd "$REPO"
mkdir -p "$OUT"

# 上游 --resume 支持自检（缺补丁则自动补上；路径必须绝对，上面已 cd 到 $REPO）
$PYTHON -c "import sys; sys.exit('--resume' not in open('train.py').read())" \
    || python3 "$WORKSPACE/recipes/xvla/train/patch_train_resume.py"

RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
    [[ -f "$RESUME/state.json" ]] || { echo "[FAIL] resume 目录缺 state.json: $RESUME"; exit 1; }
    RESUME_ARGS=(--resume "$RESUME")
    echo "Resume from: $RESUME  ($(cat "$RESUME/state.json"))"
fi

# 防重复启动：同 output_dir 已有训练在跑就直接退出，避免两个进程抢同一张卡
if pgrep -f "train\.py .*--output_dir $OUT" > /dev/null; then
    echo "[FAIL] $OUT 已有训练进程在跑，先停掉再启动：";
    pgrep -af "train\.py .*--output_dir $OUT"; exit 1
fi

echo "=== X-VLA Training: 00073 TongBot (action_mode=auto 修复版) ==="
echo "Dataset: /home/xk/dataset/00073_20260818_RM_01_yj_21 (630 eps, 194273 frames)"
echo "Model:   ./pretrained/X-VLA-Pt-auto (auto / real_dim=16 / max_dim=20)"
echo "Batch:   $BATCH_SIZE   Resume: ${RESUME:-<none, 从 0 开始>}"
echo "Output:  $OUT"

# 日志必须用 >> 追加：原脚本的 > 会把上一阶段（1~8320 步）的 train.log 截掉
nohup $PYTHON -u train.py \
    --models ./pretrained/X-VLA-Pt-auto \
    --train_metas_path /home/xk/dataset/00073_20260818_RM_01_yj_21/meta/info.json \
    --output_dir "$OUT" \
    "${RESUME_ARGS[@]}" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate 2e-4 \
    --learning_coef 0.1 \
    --iters 50000 \
    --freeze_steps 1000 \
    --warmup_steps 2000 \
    --use_cosine_decay \
    --save_interval 5000 \
    --log_interval 20 \
    >> "$OUT/train.log" 2>&1 &
TRAIN_PID=$!

echo "PID=${TRAIN_PID}"
echo "Monitor: tail -f $OUT/train.log"
# sidecar: GPU/PPU/NPU util+mem → $OUT/hw.jsonl（不碰训练；HW_MONITOR=0 关）
# shellcheck disable=SC1091
if [ -f /home/xk/data_Driven/scripts/start_hw_monitor.sh ]; then
  . /home/xk/data_Driven/scripts/start_hw_monitor.sh
  LOGDIR="$OUT" start_hw_monitor "$OUT/train.log" "${CUDA_VISIBLE_DEVICES:-0}" "${TRAIN_PID}" || true
fi
