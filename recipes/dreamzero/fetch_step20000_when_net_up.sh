#!/usr/bin/env bash
# 等 h100-103 链路恢复后自动拉取 DreamZero-14B global_step_20000 推理权重。
# 链路 2026-09-12 18:37 起断(10.1.x 网段不可达,aTrust 本体正常)。本脚本:
#   1) 轮询 gateway TCP 直到恢复(每 60s,最长 12h)
#   2) 重建 h100 ControlMaster,确认 20000 DCP 完好
#   3) 远端跑 export_dcp_model.py(CPU-only)导出 infer_weights.pt(~33G bf16)
#   4) rsync 拉回本地并双向 md5 校验
# 产物布局与 global_step_10000 一致:
#   /home/xk/VLA/h100_ckpts/dreamzero_14b/global_step_20000/actor/model_state_dict/infer_weights.pt
set -uo pipefail

H100="$HOME/bin/h100"
SCP="scp -F /home/xk/VLA/ssh_config -o ControlPath=/tmp/sshcm/%r@%h:%p"
CKPT_ROOT=/data/train_data/dreamzero-outputs/tong00073_sft_dreamzero_14b/full_full20260911k/sft/tong00073_sft_dreamzero_14b_sft/checkpoints
DCP="$CKPT_ROOT/global_step_20000/actor/dcp_checkpoint"
REMOTE_OUT=/data/train_data/dreamzero-exports/00073_step20000/infer_weights.pt
REMOTE_LOG=/data/train_data/dreamzero-exports/export_step20000.log
LOCAL_DIR=/home/xk/VLA/h100_ckpts/dreamzero_14b/global_step_20000/actor/model_state_dict
SCRIPT_SRC=/home/xk/data_Driven/recipes/dreamzero/export_dcp_model.py

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---- 1) 等网络 ----
for i in $(seq 1 720); do
  if timeout 5 nc -zv 10.1.3.52 60022 2>/dev/null; then
    log "gateway 恢复(第 ${i} 次探测)"
    break
  fi
  [ $((i % 10)) -eq 0 ] && log "仍在等待 gateway，已 $((i)) 分钟"
  sleep 60
  [ "$i" -eq 720 ] && { log "FATAL: 12h 未恢复"; exit 1; }
done

# ---- 2) 重建 master + 校验远端 ----
for t in 1 2 3; do
  $H100 true && break
  log "h100 建链失败，第 ${t} 次，退避 90s"; sleep 90
done
$H100 true || { log "FATAL: master 建不起来"; exit 2; }
$H100 "ls $DCP/.metadata" >/dev/null || { log "FATAL: 20000 DCP 缺失"; exit 3; }
log "远端 20000 DCP 完好"

# ---- 3) 远端导出(幂等:已存在则跳过)----
$H100 "test -f $REMOTE_OUT" || {
  $SCP "$SCRIPT_SRC" h100-103:/tmp/export_dcp_model.py
  mkdir_cmd="mkdir -p \$(dirname $REMOTE_OUT)"
  $H100 "$mkdir_cmd && setsid nohup /data1/rlinf-env/bin/python /tmp/export_dcp_model.py --dcp $DCP --out $REMOTE_OUT > $REMOTE_LOG 2>&1 &"
  log "远端导出已启动，轮询等待(约 10-20 分钟)"
  for i in $(seq 1 80); do
    sleep 30
    $H100 "grep -q '\[export\] wrote' $REMOTE_LOG" 2>/dev/null && break
    $H100 "grep -qiE 'Traceback|Error' $REMOTE_LOG" 2>/dev/null && {
      log "FATAL: 导出报错"; $H100 "tail -5 $REMOTE_LOG"; exit 4; }
  done
  $H100 "test -f $REMOTE_OUT" || { log "FATAL: 导出超时未完成"; exit 5; }
}
log "导出完成"

# ---- 4) rsync 拉回 + md5 ----
mkdir -p "$LOCAL_DIR"
MD5_R=$($H100 "md5sum $REMOTE_OUT" | awk '{print $1}')
log "远端 md5=$MD5_R，开始 rsync(~33G，复用 master 通道)"
rsync -avP --partial \
  -e "ssh -F /home/xk/VLA/ssh_config -o ControlPath=/tmp/sshcm/%r@%h:%p" \
  "h100-103:$REMOTE_OUT" "$LOCAL_DIR/infer_weights.pt" \
  || { log "FATAL: rsync 失败(可重跑本脚本续传)"; exit 6; }
MD5_L=$(md5sum "$LOCAL_DIR/infer_weights.pt" | awk '{print $1}')
[ "$MD5_R" = "$MD5_L" ] && log "SUCCESS: 校验一致 $MD5_L -> $LOCAL_DIR/infer_weights.pt" \
  || log "FATAL: md5 不一致 R=$MD5_R L=$MD5_L"
