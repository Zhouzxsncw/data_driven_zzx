#!/usr/bin/env bash
# h100-103 上执行:GWP 训练前数据预处理(norm stats + T5 embeddings)
# 用法: bash prep_gwp_h100.sh          (norm 用 CPU,T5 用 GPU 2)
set -uo pipefail
PY=/data1/gwp/envs/gigaworld-policy/bin/python
REPO=/data1/gwp/giga-world-policy
DS=/data1/dataset/00073_rm75_ee_v30
NORM=/data1/gwp/norm_stats_00073.json
WAN=/data1/gwp/pretrained/Wan2.2-TI2V-5B-Diffusers
log(){ echo "[$(date '+%m-%d %H:%M:%S')] prep: $*"; }

cd "$REPO"
if [ -f "$NORM" ]; then log "norm 已存在,跳过"; else
  log "1/2 compute_wam_task_norm (model_dim=16, horizon=48)"
  $PY scripts/compute_wam_task_norm.py --data-root "$DS" --output "$NORM" \
    --model-dim 16 --action-horizon 48 || { log "FATAL norm"; exit 3; }
fi

if ls "$DS/t5_embedding"/episode_000000.pt >/dev/null 2>&1; then log "t5 已存在,跳过"; else
  log "2/2 compute_t5_embedding (GPU2)"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/compute_t5_embedding.py \
    --root "$DS" --wan_path "$WAN" --device cuda --t5_folder_name t5_embedding || { log "FATAL t5"; exit 4; }
fi
log "DONE norm=$NORM t5=$(ls $DS/t5_embedding | wc -l) files"
