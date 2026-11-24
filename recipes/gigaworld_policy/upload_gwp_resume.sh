#!/usr/bin/env bash
# 续传 4/5(GWP-0.5 剩余分片) 与 5/5(Wan2.2),日志 append 进原 upload log 让 chain 能接
set -uo pipefail
SSH="ssh -F /home/xk/VLA/ssh_config -o ControlPath=/tmp/sshcm/%r@%h:%p"
LOG=/home/xk/data_Driven/recipes/gigaworld_policy/upload_gwp_h100.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
REMOTE=h100-103

up() { # $1 src/ $2 dst/  断线自动重试 4 次(rsync --partial 续传)
  local t
  for t in 1 2 3 4; do
    rsync -a --partial -e "$SSH" "$1" "$2" >> "$LOG" 2>&1 && return 0
    log "rsync 第${t}次失败,30s 后重试: $2"
    sleep 30
  done
  return 1
}

log "resume: 4/5 Giga-World-Policy-0.5(续传)"
up /data/giga-world-policy/pretrained/Giga-World-Policy-0.5/ "$REMOTE:/data1/gwp/pretrained/Giga-World-Policy-0.5/" || { log "FATAL gwp wt resume"; exit 7; }

log "resume: 5/5 Wan2.2-TI2V-5B-Diffusers"
up /data/giga-world-policy/pretrained/Wan2.2-TI2V-5B-Diffusers/ "$REMOTE:/data1/gwp/pretrained/Wan2.2-TI2V-5B-Diffusers/" || { log "FATAL wan"; exit 8; }

log "SUCCESS 全部上传完成,resume 收口"
