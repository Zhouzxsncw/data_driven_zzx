#!/usr/bin/env bash
# 看门狗:等 wait_upload_then_prep.sh 打出冒烟 READY 后,自动在 h100 拉起 GPU 2-7 六卡正式训练
set -u
R=/home/xk/data_Driven/recipes/gigaworld_policy
CHAIN=$R/h100/chain.log
SSH="ssh -F /home/xk/VLA/ssh_config -o ControlPath=/tmp/sshcm/%r@%h:%p"
LOG=$R/train_launch.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] watchdog: $*" | tee -a "$LOG"; }

BASE=$(wc -l < "$CHAIN")
log "基线 $BASE 行,等待冒烟 READY(上限 14h)"
for i in $(seq 1 840); do
  NEW=$(tail -n +"$BASE" "$CHAIN" 2>/dev/null)
  if echo "$NEW" | grep -q "FATAL"; then
    log "链路失败: $(echo "$NEW" | grep FATAL | tail -1) —— 不启动训练"; exit 1
  fi
  if echo "$NEW" | grep -q "冒烟"; then
    # 冒烟结束后再看是否标 READY;chain 若直接写 FATAL 上面已拦截
    sleep 5
    tail -n +"$BASE" "$CHAIN" | grep -q "READY" && break
  fi
  sleep 60
done
tail -n +"$BASE" "$CHAIN" | grep -q "READY" || { log "超时未见 READY,不启动"; exit 2; }

log "冒烟通过,启动六卡训练(GPU 2-7,bs8/卡,zero2,50000 步)"
$SSH h100-103 'bash /data1/gwp/local_scripts/launch_gwp_h100.sh train' >> "$LOG" 2>&1
sleep 90
if $SSH h100-103 'pgrep -f "scripts/train.py --config giga_world_policy_0_5_tongbot" | head -1' | grep -qaE "[0-9]+"; then
  log "SUCCESS 六卡训练已存活,pid=$($SSH h100-103 'pgrep -of scripts/train.py')"
else
  log "FATAL 启动后未检测到训练进程,查看 /data1/gwp/train_gwp05_00073.log"
  exit 3
fi
