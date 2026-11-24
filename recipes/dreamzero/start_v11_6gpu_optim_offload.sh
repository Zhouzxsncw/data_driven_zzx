#!/bin/bash
# V11（6 卡优化器 offload 版）：GPU 2-7 + actor.enable_offload=True（AdamW moments
# 常驻 CPU，step 时往返 GPU）。参数/梯度 GPU 常驻（不用 CPUOffloadPolicy——那会让
# AdamW 走 CPU 单核慢路径 26.75s/step，py-spy 实锤）。
# 依赖远端补丁：/data1/dreamzero-config/patch_v11_optim_offload.py（已应用）。
# resume 源：8 卡 run 的 global_step_14000。
set -euo pipefail

export MODE=full STAMP="${STAMP:-full20260911k}" GPUS=2-7 GBS=6 MAX_STEPS=20000
export EXTRA_OVERRIDES="actor.enable_offload=True"
export RESUME_DIR=/data/train_data/dreamzero-outputs/tong00073_sft_dreamzero_14b/full_full20260911a/sft/tong00073_sft_dreamzero_14b_sft/checkpoints/global_step_14000

RUN_DIR=/data/train_data/dreamzero-outputs/tong00073_sft_dreamzero_14b/full_${STAMP}
mkdir -p "${RUN_DIR}"
nohup bash /data1/dreamzero-config/launch_train_14b_h100.sh __inner_full > "${RUN_DIR}/nohup.log" 2>&1 &
echo "V11 started PID=$! RUN_DIR=${RUN_DIR}"
