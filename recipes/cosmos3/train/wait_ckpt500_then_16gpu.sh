#!/bin/bash
# 江算-16b / 16c：Cosmos3 00073 nano 现在占 0-7。
# 等 iter_000000500 写完，再停掉这路 8 卡，用同一份 checkpoint 在 0-15 上续训。
# global batch 保持 2048：32/卡 × 16 卡 × grad_accum 4。失败则按原来的 8 卡续上。
set -u
OUT=/nvme2n1/cosmos3/outputs/tongbot_00073_nano
RUN="$OUT/cosmos3_action_tongbot/action_sft/tongbot_00073_nano_8h100"
CKPT="$RUN/checkpoints"
LOG="$OUT/host_launch.log"
MARK="$OUT/wait_16gpu.state"
WAITLOG="$OUT/wait_16gpu.log"
exec >>"$WAITLOG" 2>&1
echo "===== $(date '+%F %T') waiter start ====="

if [[ -f "$MARK" && "$(cat "$MARK")" == DONE16 ]]; then
  echo "already relaunched 16gpu"
  exit 0
fi

train_pid() {
  ps -eo pid,args | awk '/torch.distributed.run/ && /action_policy_tongbot_00073_nano/ && !/awk/ {print $1; exit}'
}

has_train() {
  [[ -n "$(train_pid)" ]]
}

ckpt_ready() {
  [[ -f "$CKPT/latest_checkpoint.txt" ]] || return 1
  [[ "$(tr -d '[:space:]' < "$CKPT/latest_checkpoint.txt")" == "iter_000000500" ]] || return 1
  [[ -s "$CKPT/iter_000000500/model/.metadata" ]] || return 1
  find "$CKPT/iter_000000500/model" -name '*.distcp' | grep -q . || return 1
  grep -q 'Saved checkpoint to .*/iter_000000500' "$LOG" || return 1
  return 0
}

while ! ckpt_ready; do
  iter=$(grep -E 'iter_speed' "$LOG" 2>/dev/null | tail -1 | sed -n 's/.*] \([0-9][0-9]*\) :.*/\1/p')
  echo "$(date '+%F %T') waiting ckpt500 iter=${iter:-?}"
  sleep 60
done
echo "$(date '+%F %T') ckpt iter_000000500 ready"
sleep 20
if ! ckpt_ready; then
  echo "ckpt disappeared after wait; abort"
  exit 1
fi

pid=$(train_pid)
if [[ -z "$pid" ]]; then
  echo "no 8gpu train process; do not relaunch blindly"
  exit 1
fi
nproc=$(tr '\0' '\n' < /proc/$pid/environ | awk -F= '/^NPROC_PER_NODE=/{print $2}')
if [[ "$nproc" == "16" ]]; then
  echo "already 16 gpu pid=$pid"
  echo DONE16 > "$MARK"
  exit 0
fi

echo "$(date '+%F %T') stop 8gpu pid=$pid"
ppid=$(awk '/^PPid:/{print $2}' /proc/$pid/status)
kill -TERM "$ppid" 2>/dev/null || true
kill -TERM "$pid" 2>/dev/null || true
for _ in $(seq 1 36); do
  if ! has_train; then
    break
  fi
  sleep 5
done
if has_train; then
  echo "still alive, TERM train workers"
  ps -eo pid,args | awk '/cosmos_framework.scripts.train/ && /00073_nano/ && !/awk/ {print $1}' | while read -r w; do
    kill -TERM "$w" 2>/dev/null || true
  done
  sleep 20
fi
if has_train; then
  echo "stop failed; leave the 8gpu job running"
  exit 1
fi
echo "$(date '+%F %T') 8gpu stopped"

launch() {
  local nproc="$1" gpus="$2" port="$3" extra="$4" logfile="$5"
  docker exec -d \
    -e CUDA_VISIBLE_DEVICES="$gpus" \
    -e NPROC_PER_NODE="$nproc" \
    -e MASTER_PORT="$port" \
    -e TONGBOT_ROOT=/data/datasets/00073_20260818_RM_01_yj_21 \
    -e BASE_CHECKPOINT_PATH=/workspace/pretrained/Cosmos3-Nano-DCP \
    -e WAN_VAE_PATH=/workspace/pretrained/wan22_vae/Wan2.2_VAE.pth \
    -e OUTPUT_ROOT=/workspace/outputs/tongbot_00073_nano \
    -e IMAGINAIRE_OUTPUT_ROOT=/workspace/outputs/tongbot_00073_nano \
    -e LOG_FILENAME="$logfile" \
    -e EXTRA_TAIL_OVERRIDES="$extra" \
    -e I4_ATTN_BACKENDS= \
    -e PATH="/workspace/venv/bin:/usr/local/PPU_SDK/CUDA_SDK/bin:/usr/local/PPU_SDK/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -w /workspace/src/cosmos-framework \
    cosmos3 \
    bash examples/launch_sft_action_policy_tongbot_00073_nano.sh
}

launch 16 "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" 50016 \
  "model.config.parallelism.data_parallel_shard_degree=16 trainer.grad_accum_iter=4" \
  host_launch_16gpu.log
echo "$(date '+%F %T') launched 16gpu"
sleep 90
newpid=$(train_pid)
echo "newpid=${newpid:-none}"
if [[ -n "$newpid" ]]; then
  nohup /usr/bin/python3.11 /nvme2n1/cosmos3/code/hw/hw_train_monitor.py \
    --out "$OUT/host_launch_16gpu.log.hw.jsonl" \
    --pid "$newpid" \
    --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    >>"$OUT/host_launch_16gpu.hw_monitor.out" 2>&1 &
  echo "hw monitor $!"
fi

# 大约两步。16 卡没起来或日志里有 Traceback，就回到原来的 8 卡。
sleep 200
bad=0
if ! has_train; then
  bad=1
  echo "16gpu process gone"
fi
if grep -q 'Traceback' "$OUT/logs/host_launch_16gpu.log" 2>/dev/null; then
  bad=1
  echo "16gpu traceback"
fi
if [[ "$bad" == 1 ]]; then
  echo "$(date '+%F %T') fallback to 8gpu"
  ps -eo pid,args | awk '/cosmos_framework.scripts.train/ && /00073_nano/ {print $1}' | while read -r w; do
    kill -TERM "$w" 2>/dev/null || true
  done
  sleep 15
  launch 8 "0,1,2,3,4,5,6,7" 50012 "" host_launch_8gpu_resume.log
  echo FALLBACK8 > "$MARK"
  exit 1
fi
echo DONE16 > "$MARK"
echo "$(date '+%F %T') 16gpu still up"
