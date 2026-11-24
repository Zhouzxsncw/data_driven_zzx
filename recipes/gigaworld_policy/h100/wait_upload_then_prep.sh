#!/usr/bin/env bash
# 本机接力:等 upload_gwp_resume.sh 打点 "resume 收口" → 下发脚本 → 远端 config补丁+预处理+冒烟
# 只看基线之后的新日志行,避免被历史 FATAL/SUCCESS 干扰
set -uo pipefail
UP=/home/xk/data_Driven/recipes/gigaworld_policy/upload_gwp_h100.log
H100="$HOME/bin/h100"
SSH="ssh -F /home/xk/VLA/ssh_config -o ControlPath=/tmp/sshcm/%r@%h:%p"
HERE=/home/xk/data_Driven/recipes/gigaworld_policy/h100
log(){ echo "[$(date '+%m-%d %H:%M:%S')] chain: $*"; }

BASE=$(wc -l < "$UP")
log "基线 $BASE 行,等待 resume 收口(上限 12h)"
for i in $(seq 1 720); do
  NEW=$(tail -n +"$BASE" "$UP" 2>/dev/null)
  echo "$NEW" | grep -q "resume 收口" && break
  echo "$NEW" | grep -q "FATAL" && { log "上传失败: $(echo "$NEW" | grep FATAL | tail -1)"; exit 1; }
  sleep 60
done
echo "$(tail -n +"$BASE" "$UP" | grep -c 'resume 收口')" | grep -q "^0$" && { log "FATAL 上传超时6h"; exit 2; }

log "上传完成,下发脚本"
rsync -a -e "$SSH" "$HERE"/*.sh h100-103:/data1/gwp/local_scripts/ || { log "FATAL 下发"; exit 3; }
log "config 补丁"
$H100 'bash /data1/gwp/local_scripts/make_tongbot_config.sh' || { log "FATAL patch"; exit 3; }
log "预处理(norm+t5)"
$H100 'bash /data1/gwp/local_scripts/prep_gwp_h100.sh' > "$HERE/prep_run.log" 2>&1
if grep -q "FATAL\|Traceback" "$HERE/prep_run.log"; then log "FATAL prep: $(tail -2 "$HERE/prep_run.log" | head -1)"; exit 4; fi
log "冒烟(单卡 GPU2,15min 窗口)"
$H100 'timeout 900 bash /data1/gwp/local_scripts/launch_gwp_h100.sh smoke' > "$HERE/smoke_run.log" 2>&1
if grep -aqE "loss|iter|step" "$HERE/smoke_run.log"; then
  log "READY: 冒烟出现训练输出,可上六卡正式训练"
else
  log "WARN 冒烟无明确训练输出,人工看 smoke_run.log"
fi
