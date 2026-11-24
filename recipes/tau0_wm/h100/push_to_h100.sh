#!/usr/bin/env bash
# gpu-master → h100-103 推 τ0-WM 的开荒物料（全是小文件，1.4G 数据集不推，
# 由 h100 本地的 convert_ee_on_h100.sh 现场 FK 生成）。
#
#   bash /home/xk/data_Driven/recipes/tau0_wm/h100/push_to_h100.sh
#
# 网关现实（见 /home/xk/bin/h100 头部注释）：
#   - 频繁新建连接会被限流（kex_exchange_identification reset），所以先让 h100 建好
#     ControlMaster，所有 rsync 都复用那条通道；
#   - 长会话约 4 min 会被掐，所以每个 rsync 都带 --partial + 重试循环，断点续传。
set -uo pipefail

H100_BIN=/home/xk/bin/h100
SSH_CFG=/home/xk/VLA/ssh_config
CTL="ssh -F ${SSH_CFG} -o ControlPath=/tmp/sshcm/%r@%h:%p -o BatchMode=yes"
R=h100-103
ROOT=/data1/tau0wm
RECIPE=/home/xk/data_Driven/recipes/tau0_wm
LOG="${RECIPE}/push_to_h100.log"
: > "${LOG}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

push() {   # push <描述> <rsync 参数...>
  local desc="$1"; shift
  local i
  for i in 1 2 3 4 5; do
    if rsync -a --partial --out-format='%n %L' -e "${CTL}" "$@" >>"${LOG}" 2>&1; then
      log "OK  ${desc}（第 ${i} 次）"; return 0
    fi
    log "RETRY ${desc} 第 ${i} 次失败，20s 后重来"
    "${H100_BIN}" true >/dev/null 2>&1 || sleep 60     # 通道死了就先修
    sleep 20
  done
  log "FATAL ${desc} 五次都没推上去"; return 1
}

log "== 建链 =="
"${H100_BIN}" "echo MASTER_OK; hostname; df -h /data1 | tail -1" 2>&1 | tail -3 \
  || { log "FATAL：h100 连不上（网关限流或 VPN 掉线），稍后再跑"; exit 255; }

"${H100_BIN}" "mkdir -p ${ROOT}/{repo,local_scripts,pretrained,datasets,outputs,logs,envs}" \
  >>"${LOG}" 2>&1 || { log "FATAL 建目录失败"; exit 1; }

log "== 1/4 recipe 脚本与配置（含 EE 统计文件） =="
push "recipe"     --exclude='*.log' --exclude='*.out' --exclude='__pycache__' \
                  "${RECIPE}/" "${R}:${ROOT}/local_scripts/" || exit 1

log "== 2/4 FK 依赖：convert_tongbot_joint_to_ee.py + RM_API =="
JOINT2EE="${RECIPE}/../walloss/convert_tongbot_joint_to_ee.py"
if [[ ! -f "${JOINT2EE}" ]]; then
  JOINT2EE=/home/xk/VLA/wall_oss_ppu_setup/scripts/convert_tongbot_joint_to_ee.py
fi
push "joint2ee"   "${JOINT2EE}" \
                  "${R}:${ROOT}/local_scripts/" || exit 1
push "RM_API"     --exclude='__pycache__' --exclude='win_32' --exclude='win_64' \
                  /home/xk/RM_API/ "${R}:${ROOT}/local_scripts/RM_API/" || exit 1

log "== 3/4 τ0-WM 仓库（代码 19M，不含权重） =="
push "repo"       --exclude='.git' --exclude='__pycache__' --exclude='outputs' \
                  /home/xk/VLA/tau-0-wm/ "${R}:${ROOT}/repo/tau-0-wm/" || exit 1

log "== 4/4 落地校验 =="
"${H100_BIN}" "cd ${ROOT}/local_scripts && ls -l patch_tau_repo.py convert_00036_ee.py \
    convert_tongbot_joint_to_ee.py configs/stat_file/tongbot_00036_ee_rela.json h100/*.sh && \
    du -sh ${ROOT}/repo/tau-0-wm ${ROOT}/local_scripts/RM_API && \
    test -f ${ROOT}/repo/tau-0-wm/main.py && echo PUSH_OK" 2>&1 | tail -12 | tee -a "${LOG}"

log "下一步（都在 h100 上后台跑，互不依赖的可以并行）："
log "  1) setsid nohup bash ${ROOT}/local_scripts/h100/setup_h100.sh > ${ROOT}/setup_h100.out 2>&1 &"
log "  2) 环境好之后：setsid nohup bash ${ROOT}/local_scripts/h100/convert_ee_on_h100.sh > ${ROOT}/convert_ee.out 2>&1 &"
log "  3) 最后挂接力：setsid nohup bash ${ROOT}/local_scripts/h100/relay_watch.sh > ${ROOT}/relay.out 2>&1 &"
