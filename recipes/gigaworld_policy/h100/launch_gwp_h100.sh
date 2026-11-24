#!/usr/bin/env bash
# h100-103 上启动 GigaWorld-Policy-0.5 00073 微调(GPU 2-7 六卡 DeepSpeed zero2)
# 用法:
#   bash launch_gwp_h100.sh smoke    # 单卡(GPU2)bs1 冒烟,验证全链路后自动退出
#   bash launch_gwp_h100.sh train    # 六卡正式训练(nohup 后台)
set -uo pipefail
MODE=${1:-smoke}
PY=/data1/gwp/envs/gigaworld-policy/bin/python
# Launcher 内部用 `which accelerate` 定位启动器,必须把 env/bin 挂进 PATH
export PATH=/data1/gwp/envs/gigaworld-policy/bin:$PATH
REPO=/data1/gwp/giga-world-policy
DS=/data1/dataset/00073_rm75_ee_v30
OUT=/data1/gwp/outputs/00073_gwp05_ft

export GWP_AGILEX_DATA_PATHS="$DS"
export GWP_NORM_STATS_PATH=/data1/gwp/norm_stats_00073.json
export GWP_WAN_PRETRAINED=/data1/gwp/pretrained/Wan2.2-TI2V-5B-Diffusers
export GWP05_TRANSFORMER_PRETRAINED=/data1/gwp/pretrained/Giga-World-Policy-0.5
export GWP_T5_LOAD_FROM=dir
export GWP05_PROJECT_DIR=$OUT
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

cd "$REPO"
CONFIG=giga_world_policy_0_5_tongbot_finetune

if [ "$MODE" = "smoke" ]; then
  WA_DEBUG_SINGLE_GPU=1 CUDA_VISIBLE_DEVICES=2 GWP_GPU_IDS=0 \
    GWP_BATCH_SIZE_PER_GPU=1 GWP_NUM_WORKERS=2 \
    $PY scripts/train.py --config "$CONFIG" 2>&1 | tee /data1/gwp/smoke.log
  exit $?
fi

# 正式:六卡。bs8+EMA 实测 OOM(12B 模型 zero2 每卡权重24G+EMA24G),
# 改 bs4 + 关 EMA(config 补丁)+ expandable_segments 抗碎片;跑稳后再回调 bs
# GWP_BS 控每卡 batch(默认 4);GWP_RESUME=1 时把 config 的 resume 翻 True 并续训
GWP_BS=${GWP_BS:-4}
GWP_LOG=${GWP_LOG:-/data1/gwp/train_gwp05_00073.log}
if [ "${GWP_RESUME:-0}" = "1" ]; then
  grep -q 'resume=True' configs/${CONFIG}.py || \
    sed -i 's/resume=False/resume=True/' configs/${CONFIG}.py
  echo "$(date '+%F %T') launch: 已置 resume=True -> $(grep -nE 'resume=' configs/${CONFIG}.py)"
fi
GWP_GPU_IDS=2,3,4,5,6,7 \
  GWP_BATCH_SIZE_PER_GPU=$GWP_BS GWP_NUM_WORKERS=6 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup $PY scripts/train.py --config "$CONFIG" \
  > "$GWP_LOG" 2>&1 &
echo "GWP train PID=$! bs=$GWP_BS log=$GWP_LOG"
