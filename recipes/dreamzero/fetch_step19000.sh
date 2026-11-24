#!/usr/bin/env bash
# 等 h100 上 step_19000 导出完成 → rsync 拉回 → md5 校验
set -uo pipefail
H100="$HOME/bin/h100"
REMOTE_OUT=/data/train_data/dreamzero-exports/00073_step19000/infer_weights.pt
REMOTE_LOG=/data/train_data/dreamzero-exports/export_step19000.log
LOCAL_DIR=/home/xk/VLA/h100_ckpts/dreamzero_14b/global_step_19000/actor/model_state_dict
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$LOCAL_DIR"
log "等待远端导出(1317 tensors dcp.load→bf16)"
for i in $(seq 1 120); do
  sleep 30
  $H100 "grep -q '\[export\] wrote' $REMOTE_LOG" 2>/dev/null && break
  $H100 "grep -qiE 'Traceback' $REMOTE_LOG" 2>/dev/null && { log "FATAL 导出报错:"; $H100 "tail -4 $REMOTE_LOG"; exit 4; }
done
$H100 "test -f $REMOTE_OUT" || { log "FATAL 导出超时"; exit 5; }
MD5_R=$($H100 "md5sum $REMOTE_OUT" | awk '{print $1}')
log "导出完成 md5=$MD5_R,开始 rsync"
rsync -avP --partial -e "ssh -F /home/xk/VLA/ssh_config -o ControlPath=/tmp/sshcm/%r@%h:%p" \
  "h100-103:$REMOTE_OUT" "$LOCAL_DIR/infer_weights.pt" || { log "FATAL rsync 失败(重跑续传)"; exit 6; }
MD5_L=$(md5sum "$LOCAL_DIR/infer_weights.pt" | awk '{print $1}')
[ "$MD5_R" = "$MD5_L" ] && log "SUCCESS step19000 校验一致 $MD5_L" || log "FATAL md5 不一致"
