#!/usr/bin/env bash
# 「等通道恢复再推」的外层循环：gateway(10.1.3.52:60022) 对新建连接限流，
# 表现为 TCP 通但 kex_exchange_identification: Connection reset by peer。
# 一次失败后连续重试只会把冷却时间越拖越长（实测重度使用惩罚 15min+），
# 所以这里固定「一次只试一个 master 握手 + 间隔 5 分钟」，最多 24 次（2 h）。
#
#   setsid nohup bash /home/xk/data_Driven/recipes/tau0_wm/h100/push_when_up.sh \
#       > /home/xk/data_Driven/recipes/tau0_wm/push_when_up.out 2>&1 &
set -uo pipefail

SSH_CFG=/home/xk/VLA/ssh_config
CTL_DIR=/tmp/sshcm
ASKPASS="$HOME/.ssh/h100_askpass.sh"
LOG=/home/xk/data_Driven/recipes/tau0_wm/push_when_up.log
INTERVAL="${INTERVAL:-300}"
MAX_TRIES="${MAX_TRIES:-24}"
export BACKOFF_S="${BACKOFF_S:-300}"      # 被 push_to_h100.sh 里的 h100 调用继承
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${CTL_DIR}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

master_alive() { ssh -F "${SSH_CFG}" -o "ControlPath=${CTL_DIR}/%r@%h:%p" -O check h100-103 &>/dev/null; }

try_master() {   # 单次握手（成功则 master 后台常驻 2h）
  local sock
  sock="$(ssh -G -F "${SSH_CFG}" h100-103 2>/dev/null \
          | awk '/^hostname /{h=$2} /^user /{u=$2} /^port /{p=$2} END{printf "%s/%s@%s:%s}", d, u, h, p}' d="${CTL_DIR}")"
  [[ -S "${sock}" ]] && rm -f "${sock}"
  SSH_ASKPASS="${ASKPASS}" SSH_ASKPASS_REQUIRE=force setsid ssh \
    -F "${SSH_CFG}" -o "ControlPath=${CTL_DIR}/%r@%h:%p" \
    -o ControlMaster=yes -o ControlPersist=2h -o ConnectTimeout=30 \
    -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
    h100-103 true </dev/null >/dev/null 2>&1 &
  local i
  for i in $(seq 1 40); do master_alive && return 0; sleep 1; done
  return 1
}

for n in $(seq 1 "${MAX_TRIES}"); do
  if master_alive; then
    log "第 ${n} 次：复用已有 master"
  elif try_master; then
    log "第 ${n} 次：master 建好了"
  else
    log "第 ${n} 次：握手失败（gateway 限流），${INTERVAL}s 后再试"
    sleep "${INTERVAL}"
    continue
  fi
  log "开始推送 → push_to_h100.sh"
  if bash "${HERE}/push_to_h100.sh"; then
    log "PUSH_WHEN_UP_OK"
    exit 0
  fi
  log "推送失败，通道可能又死了；${INTERVAL}s 后重来"
  sleep "${INTERVAL}"
done
log "PUSH_WHEN_UP_FATAL：${MAX_TRIES} 次都没能建立 h100 通道（多半是 VPN 掉了或 gateway 长期拉黑）"
exit 3
