#!/usr/bin/env bash
# 并行加流:专传 Wan2.2-TI2V-5B-Diffusers(与主 resume 的 GWP 流互不干扰)
set -uo pipefail
SSH="ssh -F /home/xk/VLA/ssh_config -o ControlPath=/tmp/sshcm/%r@%h:%p"
LOG=/home/xk/data_Driven/recipes/gigaworld_policy/upload_gwp_h100.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] wan-parallel: $*" >> "$LOG"; }
log "加流启动: Wan2.2 → /data1/gwp/pretrained/Wan2.2-TI2V-5B-Diffusers/"
t=0
until rsync -a --partial -e "$SSH" /data/giga-world-policy/pretrained/Wan2.2-TI2V-5B-Diffusers/ "h100-103:/data1/gwp/pretrained/Wan2.2-TI2V-5B-Diffusers/" >> "$LOG" 2>&1; do
  t=$((t+1)); log "Wan 流第${t}次断,45s 后重试"; sleep 45
  [ $t -ge 30 ] && { log "FATAL wan 30连败"; exit 8; }
done
log "Wan 流完成(主脚本 5/5 将秒过)"
