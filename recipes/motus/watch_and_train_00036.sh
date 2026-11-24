#!/usr/bin/env bash
# gpu-node1: 等 FastWAM 00073 训练结束并释放 GPU 后，自动启动 Motus 00036 训练。
# 用法: nohup bash /data/motus/watch_and_train_00036.sh >/dev/null 2>&1 &
# 日志: /data/motus/watch_and_train.log（watcher） + /data/motus/train_00036.log（训练）
set -u

WATCH_LOG=/data/motus/watch_and_train.log
TRAIN_LOG=/data/motus/train_00036.log

log() { echo "[$(date '+%F %T')] $*" >> "${WATCH_LOG}"; }

log "===== watcher started (pid $$) ====="
log "waiting for FastWAM tongbot_00073 train.py to exit (check every 5 min)"

# 1) 等 FastWAM 训练进程退出
while pgrep -f 'scripts/train.py.*tongbot_00073' > /dev/null 2>&1; do
  sleep 300
done
log "FastWAM train.py exited"

# 2) 等 GPU 显存释放（< 8 GB）
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${used}" -lt 8000 ]; then
    break
  fi
  sleep 60
done
log "GPU free (used=${used} MiB), starting Motus 00036"

# 3) 启动 Motus 00036 训练
export MOTUS_ROOT=/data/motus/Motus
source /home/cqcigai/anaconda3/etc/profile.d/conda.sh
conda activate motus
cd /data/motus/Motus

NPROC=1 \
CONFIG_FILE=configs/tongbot_00036.yaml \
TASK=tongbot_00036 \
OUTPUT_DIR=/data/motus/outputs/00036/run_$(date +%Y%m%d_%H%M%S) \
  bash /data/motus/recipes/launch_train.sh >> "${TRAIN_LOG}" 2>&1
rc=$?
log "Motus training process exited rc=${rc}"

# 4) 结果检查
if grep -qi 'out of memory\|CUDA out of memory' "${TRAIN_LOG}" 2>/dev/null; then
  log "WARNING: OOM detected in train log — consider reducing batch_size or enabling CPU offload"
fi
log "===== watcher done ====="
