#!/bin/bash
# 江算-16b / 16c：当前 16 卡 Cosmos 00073 自己跑到 max_iter=3000 后退出。
# 本脚本不杀进程。等到 iter_000003000 写完，再用同一实验目录把 max_iter 改成 8000 续上。
# 中途崩溃、没有 3000 档，就不续。
set -u
OUT=/nvme2n1/cosmos3/outputs/tongbot_00073_nano
RUN="$OUT/cosmos3_action_tongbot/action_sft/tongbot_00073_nano_8h100"
CKPT="$RUN/checkpoints"
MARK="$OUT/wait_max8000.state"
WAITLOG="$OUT/wait_max8000.log"
LOCK="$OUT/wait_max8000.lock"
exec >>"$WAITLOG" 2>&1
exec 9>"$LOCK"
flock -n 9 || { echo "$(date '+%F %T') another waiter holds the lock"; exit 0; }

echo "===== $(date '+%F %T') waiter start pid=$$ ====="

if [[ -f "$MARK" && "$(cat "$MARK")" == DONE8000 ]]; then
  echo "already resumed to max_iter=8000"
  exit 0
fi
echo WAITING > "$MARK"

train_pid() {
  ps -eo pid,args | awk '/torch.distributed.run/ && /action_policy_tongbot_00073_nano/ && !/awk/ {print $1; exit}'
}

ckpt_3000_ready() {
  [[ -f "$CKPT/latest_checkpoint.txt" ]] || return 1
  [[ "$(tr -d '[:space:]' < "$CKPT/latest_checkpoint.txt")" == "iter_000003000" ]] || return 1
  [[ -s "$CKPT/iter_000003000/model/.metadata" ]] || return 1
  find "$CKPT/iter_000003000/model" -name '*.distcp' | grep -q . || return 1
  return 0
}

while pid=$(train_pid); [[ -n "$pid" ]]; do
  echo "$(date '+%F %T') still training pid=$pid"
  sleep 60
done
echo "$(date '+%F %T') train process gone"

ready=0
for _ in $(seq 1 40); do
  if ckpt_3000_ready; then
    ready=1
    break
  fi
  latest=$(tr -d '[:space:]' < "$CKPT/latest_checkpoint.txt" 2>/dev/null || true)
  echo "$(date '+%F %T') waiting iter_000003000 latest=${latest:-none}"
  sleep 60
done
if [[ "$ready" != 1 ]]; then
  echo "$(date '+%F %T') ABORT: ended without a complete iter_000003000. not resuming."
  echo ABORT > "$MARK"
  exit 1
fi
sleep 20
if ! ckpt_3000_ready; then
  echo "iter_000003000 disappeared after wait; abort"
  echo ABORT > "$MARK"
  exit 1
fi
if [[ -n "$(train_pid)" ]]; then
  echo "a train process appeared before relaunch; leave it"
  exit 0
fi

echo "$(date '+%F %T') launching max_iter=8000 from iter_000003000"
docker exec -d \
  -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
  -e NPROC_PER_NODE="16" \
  -e MASTER_PORT="50016" \
  -e TONGBOT_ROOT=/data/datasets/00073_20260818_RM_01_yj_21 \
  -e BASE_CHECKPOINT_PATH=/workspace/pretrained/Cosmos3-Nano-DCP \
  -e WAN_VAE_PATH=/workspace/pretrained/wan22_vae/Wan2.2_VAE.pth \
  -e OUTPUT_ROOT=/workspace/outputs/tongbot_00073_nano \
  -e IMAGINAIRE_OUTPUT_ROOT=/workspace/outputs/tongbot_00073_nano \
  -e LOG_FILENAME="host_launch_16gpu_flash_8k.log" \
  -e COSMOS3_MODEL=/workspace/pretrained/Cosmos3-Nano \
  -e EXTRA_TAIL_OVERRIDES="model.config.parallelism.data_parallel_shard_degree=16 trainer.grad_accum_iter=1 trainer.max_iter=8000 dataloader_train.max_samples_per_batch=128" \
  -e I4_ATTN_BACKENDS= \
  -e PATH="/workspace/venv/bin:/usr/local/PPU_SDK/CUDA_SDK/bin:/usr/local/PPU_SDK/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -w /workspace/src/cosmos-framework \
  cosmos3 \
  bash examples/launch_sft_action_policy_tongbot_00073_nano.sh

sleep 90
newpid=$(train_pid)
echo "newpid=${newpid:-none}"
if [[ -z "$newpid" ]]; then
  echo "$(date '+%F %T') launch failed; train process not found"
  echo ABORT > "$MARK"
  exit 1
fi
if ! tr '\0' '\n' < /proc/"$newpid"/environ | grep -q 'trainer.grad_accum_iter=1'; then
  echo "new process missing grad_accum_iter=1; leave it running and mark ABORT for a human"
  echo ABORT > "$MARK"
  exit 1
fi
nohup /usr/bin/python3.11 /nvme2n1/cosmos3/code/hw/hw_train_monitor.py \
  --out "$OUT/logs/host_launch_16gpu_flash_8k.log.hw.jsonl" \
  --pid "$newpid" \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  >>"$OUT/host_launch_16gpu_flash_8k.hw_monitor.out" 2>&1 &
echo "hw monitor $!"
echo DONE8000 > "$MARK"
echo "$(date '+%F %T') resumed pid=$newpid max_iter=8000"
