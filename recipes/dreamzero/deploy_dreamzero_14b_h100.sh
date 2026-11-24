#!/usr/bin/env bash
# ============================================================
# 本机执行：把 DreamZero-14B 的 H100 训练配置推到 h100-103 并远端校验
#
# 用法:
#   bash deploy_dreamzero_14b_h100.sh              # 推送 + 远端 MODE=check（preflight，不占卡）
#   ACTION=smoke  bash deploy_dreamzero_14b_h100.sh  # 推送 + 校验 + 4 步冒烟（要 8 卡空）
#   ACTION=full   bash deploy_dreamzero_14b_h100.sh  # 推送 + 校验 + 冒烟 + 后台正式训练
#   ACTION=push   bash deploy_dreamzero_14b_h100.sh  # 只推文件，不跑远端
#   ACTION=remote bash deploy_dreamzero_14b_h100.sh  # 只跑远端 check（不推）
#
# 只推小文本（配置/脚本/metadata），权重和代码树一律不动：
# 网关内网实测只有 ~0.76MB/s，大文件必须在 H100 本地从 ModelScope 下。
# ============================================================
set -euo pipefail

: "${SSHPASS:?Set SSHPASS}"
export SSHPASS
HOST="${HOST:-h100-103}"
SSH_OPTS="-F /home/xk/VLA/ssh_config -o StrictHostKeyChecking=no -o ConnectTimeout=90 -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
ACTION="${ACTION:-check}"                 # push | check | smoke | full | remote
REPO="${REPO:-/home/xk/data_Driven}"
RLINF_LOCAL="${RLINF_LOCAL:-/home/xk/VLA/RLinf}"
STAGING="${STAGING:-/home/xk/VLA/dreamzero_staging}"

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "FATAL: $*"; exit 1; }

rssh() {                     # 远端执行（网关 kex 会被限流，失败自动退避重试）
  local tries="${TRIES:-4}" i
  for i in $(seq 1 "${tries}"); do
    if timeout 1800 sshpass -e ssh ${SSH_OPTS} "${HOST}" "$@" 2>&1; then
      return 0
    fi
    log "  ssh 第 ${i}/${tries} 次失败，${i}0s 后重试（网关可能在限流）"
    sleep $(( i * 10 ))
  done
  return 1
}

# src(本机) -> dst(H100 绝对路径)；md5 相同就跳过，可重复执行
FILES=(
  "${REPO}/recipes/dreamzero/tong00073_sft_dreamzero_14b.yaml|/data1/RLinf/examples/sft/config/tong00073_sft_dreamzero_14b.yaml"
  "${RLINF_LOCAL}/examples/sft/config/model/dreamzero_14b.yaml|/data1/RLinf/examples/sft/config/model/dreamzero_14b.yaml"
  "${REPO}/recipes/dreamzero/launch_train_14b_h100.sh|/data1/dreamzero-config/launch_train_14b_h100.sh"
  "${REPO}/recipes/dreamzero/check_sft_config.py|/data1/dreamzero-config/check_sft_config.py"
  "${STAGING}/metadata_00073.json|/data1/dreamzero-config/metadata_00073.json"
  "${STAGING}/download_14b_h100.sh|/data1/dreamzero-config/download_14b_h100.sh"
)

push_all() {
  local entry src dst lmd rmd
  for entry in "${FILES[@]}"; do
    src="${entry%%|*}"; dst="${entry##*|}"
    [ -f "${src}" ] || die "本机缺文件 ${src}"
    lmd=$(md5sum "${src}" | cut -c1-12)
    rmd=$(rssh "mkdir -p \"\$(dirname '${dst}')\" && md5sum '${dst}' 2>/dev/null | cut -c1-12" | tail -1)
    if [ "${rmd}" = "${lmd}" ]; then
      log "SKIP  $(basename "${dst}")（md5 ${lmd} 已一致）"
      continue
    fi
    log "PUSH  $(basename "${src}") -> ${dst}  [${rmd:- absent} -> ${lmd}]"
    sshpass -e ssh ${SSH_OPTS} "${HOST}" "cat > '${dst}' && chmod +x '${dst}'" < "${src}" \
      || die "推送失败 ${dst}"
    [ "$(rssh "md5sum '${dst}' 2>/dev/null | cut -c1-12" | tail -1)" = "${lmd}" ] \
      || die "推送后 md5 不符：${dst}"
  done
  log "文件推送完成（共 ${#FILES[@]} 项，均已校验）"
}

remote_info() {
  log "---------- 远端现场（代码版本 / 14B 资产 / 卡 / 盘） ----------"
  rssh 'set -x
    for d in /data1/RLinf /data1/dreamzero; do
      printf "%s HEAD=" "$d"; git -C "$d" rev-parse --short HEAD 2>/dev/null || echo "(no git)"
    done
    ls -l /data1/dreamzero-pretrained/Wan2.1-I2V-14B-480P/ 2>/dev/null | head -14
    du -sh /data1/dreamzero-pretrained/* 2>/dev/null
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    df -BG --output=target,avail /data /data1 | tail -3'
}

remote_run() {   # $1 = check | smoke | full
  local mode="$1"
  log "---------- 远端 MODE=${mode} ----------"
  # 训练入口内部会 source venv，这里只需给一个能用的 login shell
  rssh "cd /data1 && MODE=${mode} bash /data1/dreamzero-config/launch_train_14b_h100.sh"
}

log "DreamZero-14B deploy — ACTION=${ACTION} host=${HOST}"
log "本机配置: ${REPO}/recipes/dreamzero/"

case "${ACTION}" in
  push)
    push_all
    ;;
  remote)
    remote_info
    remote_run check
    ;;
  check)
    push_all
    remote_info
    remote_run check
    log "完成。8 卡冒烟： ACTION=smoke bash $0 ；正式训练： ACTION=full bash $0"
    ;;
  smoke)
    push_all
    remote_run smoke
    ;;
  full)
    push_all
    remote_run full
    ;;
  *)
    die "未知 ACTION=${ACTION}（push|check|smoke|full|remote）"
    ;;
esac
