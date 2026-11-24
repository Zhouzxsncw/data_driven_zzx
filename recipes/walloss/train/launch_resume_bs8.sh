#!/bin/bash
set -euo pipefail
# 不要 source envsetup.sh：它会 exit 1 把整个启动脚本杀掉。
source /workspace/venv/bin/activate
export CUDA_VISIBLE_DEVICES=1,2,3,4
export CHECKPOINT_OPTIMIZATION=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export WALLX_DIST_TIMEOUT_MINUTES=60
export NCCL_DEBUG=WARN
LOGDIR=/workspace/outputs/walloss05_tongbot_ee_00073
mkdir -p "$LOGDIR"
echo "[resume-bs8] $(date -Is) nproc=4 cfg=sft_tongbot_ee_00073_4gpu_resume_bs8.yml" >> "$LOGDIR/launch.log"
cd /workspace/src/wall-x
# sidecar: GPU/PPU/NPU util+mem → *.hw.jsonl（不碰训练；HW_MONITOR=0 关）
# shellcheck disable=SC1091
for _hw_src in \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_hw_monitor.sh" \
  /home/xk/data_Driven/scripts/start_hw_monitor.sh \
  /workspace/code/train/start_hw_monitor.sh \
  /nvme2n1/wall-oss/code/train/start_hw_monitor.sh; do
  [ -f "$_hw_src" ] || continue
  . "$_hw_src"
  break
done
unset _hw_src
if declare -F start_hw_monitor >/dev/null; then
  start_hw_monitor "${LOGDIR}/nohup.out" || true
fi
exec python -m torch.distributed.run --nproc_per_node=4 --master_port=29618 \
  /workspace/code/train/run_sft.py \
  --config /workspace/code/train/sft_tongbot_ee_00073_4gpu_resume_bs8.yml \
  --log_to_file
