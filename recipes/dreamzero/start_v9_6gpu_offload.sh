#!/bin/bash
# V9（6 卡 offload 提速版）：GPU 2-7 + FSDP2 cpu_offload + pin memory + 关 prefetch
# 背景：6 卡无 offload 物理不够（分片状态比 8 卡多 9G/卡，恒定 79.06G OOM，已试
# V2/V3/V4/V7/V7b 五个组合验证）；V5（cpu_offload 无 pin、prefetch 开）能跑但
# 30s/step。V9 三个提速点：offload_pin_memory=True（DMA 直传输）、
# forward_prefetch=False + backward_prefetch=null（offload 下 prefetch 触发频繁
# 小传输且与逐层 offload 交互差）。
# resume 源：8 卡 run 的 global_step_14000。
set -euo pipefail

export MODE=full STAMP="${STAMP:-full20260911i}" GPUS=2-7 GBS=6 MAX_STEPS=20000
export EXTRA_OVERRIDES="actor.fsdp_config.cpu_offload=True actor.fsdp_config.offload_pin_memory=True actor.fsdp_config.forward_prefetch=False actor.fsdp_config.backward_prefetch=null"
export RESUME_DIR=/data/train_data/dreamzero-outputs/tong00073_sft_dreamzero_14b/full_full20260911a/sft/tong00073_sft_dreamzero_14b_sft/checkpoints/global_step_14000

RUN_DIR=/data/train_data/dreamzero-outputs/tong00073_sft_dreamzero_14b/full_${STAMP}
mkdir -p "${RUN_DIR}"
nohup bash /data1/dreamzero-config/launch_train_14b_h100.sh __inner_full > "${RUN_DIR}/nohup.log" 2>&1 &
echo "V9 started PID=$! RUN_DIR=${RUN_DIR}"
