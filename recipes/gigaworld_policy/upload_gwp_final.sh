#!/usr/bin/env bash
# GWP 上传(append-verify 版):网关 ~4min 掐线 + rsync 随机后缀临时文件不可复用,
# 必须 --append-verify 直接在目标文件上续传,否则每次重连从 0 开始
set -u
SSH="ssh -F /home/xk/VLA/ssh_config -o ControlPath=/tmp/sshcm/%r@%h:%p"
LOG=/home/xk/data_Driven/recipes/gigaworld_policy/upload_gwp_h100.log
SRC=/data/giga-world-policy/pretrained/Giga-World-Policy-0.5
DST=h100-103:/data1/gwp/pretrained/Giga-World-Policy-0.5
log() { echo "[$(date '+%m-%d %H:%M:%S')] final: $*" >> "$LOG"; }
for f in diffusion_pytorch_model-00002-of-00003.safetensors \
         diffusion_pytorch_model-00003-of-00003.safetensors \
         diffusion_pytorch_model.safetensors.index.json; do
  log "传 $f"
  n=0
  until rsync --append-verify --partial --timeout=60 -e "$SSH" "$SRC/$f" "$DST/" >> "$LOG" 2>&1; do
    n=$((n+1))
    cur=$(~/bin/h100 "stat -c %s $DST/$f 2>/dev/null || echo 0")
    log "$f 第${n}次断,已传 ${cur} 字节,60s 后续传"
    [ $n -ge 60 ] && { log "FATAL $f"; exit 1; }
    sleep 60
  done
  log "$f DONE"
done
log "resume 收口"
