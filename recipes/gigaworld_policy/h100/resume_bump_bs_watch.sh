#!/usr/bin/env bash
# 等 step_10000 的 ckpt 完整落盘 -> kill 旧六卡(bs4,非 resume)-> 以 bs8 + resume=True 重启
# 用法(在 h100 上后台跑): setsid nohup bash resume_bump_bs_watch.sh >/data1/gwp/watch_bump.log 2>&1 &
set -uo pipefail
CKROOT=/data1/gwp/outputs/00073_gwp05_ft/models
CFG=/data1/gwp/giga-world-policy/configs/giga_world_policy_0_5_tongbot_finetune.py
LAUNCH=/data1/gwp/local_scripts/launch_gwp_h100.sh
TRAINLOG=/data1/gwp/train_gwp05_00073.log
BS=8
W=/data1/gwp/bump_resume.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] bump: $*" | tee -a "$W"; }

say "watcher 启动,目标 bs=$BS,等待 $CKROOT 下出现完整 checkpoint_*_step_10000"

# 记录基线,确保抓到的是本次新存的 ckpt
BASE_LINES=$(wc -l < "$TRAINLOG" 2>/dev/null || echo 0)
CK=""
for i in $(seq 1 200); do                 # 上限 ~200*20s ≈ 67min
  cand=$(ls -d "$CKROOT"/checkpoint_*_step_10000 2>/dev/null | sort -t_ -k5 -n | tail -1)
  if [ -n "$cand" ] && [ -d "$cand/pytorch_model" ]; then
    # 稳定性:连续 3 次(每次 15s)目录内无新写入,认为 DeepSpeed 已写完
    stable=0
    for k in 1 2 3; do
      sleep 15
      fresh=$(find "$cand" -newermt '-15 seconds' 2>/dev/null | head -1)
      if [ -z "$fresh" ]; then stable=$((stable+1)); else stable=0; fi
    done
    if [ "$stable" -ge 3 ]; then CK="$cand"; break; fi
    say "ckpt 目录出现但未写完,继续等(轮 $i)"
  else
    # 旧进程若已不在,异常退出
    pgrep -f 'scripts/train.py --config giga_world_policy_0_5_tongbot' >/dev/null \
      || { say "FATAL: 训练进程已消失但未见 ckpt"; exit 5; }
    [ $((i % 9)) -eq 0 ] && say "轮 $i: 尚无 step_10000 ckpt,继续等"
    sleep 20
  fi
done
[ -z "$CK" ] && { say "FATAL: 超时未见完整 ckpt"; exit 6; }
say "ckpt 就绪: $CK  (size=$(du -sh "$CK" 2>/dev/null | cut -f1))"

say "kill 旧训练(setsid 进程组)"
pids=$(pgrep -f 'scripts/train.py --config giga_world_policy_0_5_tongbot' | tr '\n' ' ')
say "旧 PID: $pids"
for p in $pids; do
  pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
  [ -n "$pg" ] && kill -TERM -"$pg" 2>/dev/null
done
sleep 25
pkill -9 -f 'scripts/train.py --config giga_world_policy_0_5_tongbot' 2>/dev/null
pkill -9 -f 'run_task.py --config configs.giga_world_policy_0_5_tongbot' 2>/dev/null
say "等待 GPU 显存释放..."
for t in $(seq 1 20); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '3,8p' | sort -n | tail -1)
  say "GPU2-7 峰值占用 ${used} MiB"
  [ "${used:-99999}" -lt 3000 ] && break
  sleep 10
done

say "bs=$BS resume 重启六卡"
GWP_BS=$BS GWP_RESUME=1 GWP_LOG=/data1/gwp/train_gwp05_00073_bs8.log \
  bash "$LAUNCH" train | tee -a "$W"
say "已下发启动,等待首步确认 resume 生效..."
sleep 120
tail -30 /data1/gwp/train_gwp05_00073_bs8.log 2>/dev/null | tee -a "$W"
say "bump 完成,新日志 /data1/gwp/train_gwp05_00073_bs8.log"
