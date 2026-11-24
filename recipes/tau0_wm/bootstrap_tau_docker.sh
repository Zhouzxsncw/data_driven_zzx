#!/bin/bash
# 本机跑：给训练机新开独立 tau0wm 容器并装依赖。
# 不做：江算-1 / 江算-4 / gpu-master / 推理电脑 / 江算-16 原机 / gpu-node1 / 松龄 4090。
# 不杀训练、不删已有容器。
#
#   bash recipes/tau0_wm/bootstrap_tau_docker.sh
#   bash recipes/tau0_wm/bootstrap_tau_docker.sh 16b cty-18 a3
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="${LOCAL_REPO:-/home/xk/VLA/tau-0-wm}"
mkdir -p /tmp/sshcm-js /tmp/sshcm-hw

JS_SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15
  -o ControlMaster=auto -o ControlPath=/tmp/sshcm-js/%r@%h:%p -o ControlPersist=30m)
HW_SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15
  -o ControlMaster=auto -o ControlPath=/tmp/sshcm-hw/%r@%h:%p -o ControlPersist=30m)
export SSHPASS="${HUAWEI_PW:-$(grep -hoP "PW='\K[^']+" /home/xk/.pull_00073_ckpts_v2.sh | head -1)}"
A2_SSH=(sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=15
  -o PreferredAuthentications=password -o PubkeyAuthentication=no)

RSYNC_EX=(--exclude '.git' --exclude '__pycache__' --exclude '*.jsonl' --exclude '*.log'
  --exclude 'push_when_up.out')

if [[ ! -d "${LOCAL_REPO}/runner" ]]; then
  echo "[ERR] 缺本地仓库 ${LOCAL_REPO}" >&2
  exit 1
fi

sync_one() {
  local label="$1" dest="$2"
  shift 2
  echo "===== sync ${label} -> ${dest} ====="
  "$@" "mkdir -p ${dest}/code ${dest}/repo ${dest}/pretrained ${dest}/logs ${dest}/shim ${dest}/outputs ${dest}/hf"
  rsync -az "${RSYNC_EX[@]}" -e "${JS_PLACEHOLDER}" >/dev/null
}

# rsync -e needs a string. Build per target below.

push_js() {
  local host="$1" dest="$2"
  local e="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=/tmp/sshcm-js/%r@%h:%p -o ControlPersist=30m"
  "${JS_SSH[@]}" "${host}" "mkdir -p ${dest}/code ${dest}/repo ${dest}/pretrained ${dest}/logs ${dest}/shim ${dest}/outputs ${dest}/hf"
  rsync -az "${RSYNC_EX[@]}" -e "${e}" "${HERE}/" "${host}:${dest}/code/"
  rsync -az --exclude '.git' --exclude '__pycache__' -e "${e}" "${LOCAL_REPO}/" "${host}:${dest}/repo/tau-0-wm/"
}

push_h100() {
  local dest="/data1/tau0wm"
  local e="ssh -F ${HOME}/VLA/ssh_config -o BatchMode=yes -o ControlPath=/tmp/sshcm/%r@%h:%p"
  h100 "mkdir -p ${dest}/code ${dest}/repo ${dest}/pretrained ${dest}/logs ${dest}/shim"
  rsync -az "${RSYNC_EX[@]}" -e "${e}" "${HERE}/" "h100-103:${dest}/code/"
  rsync -az --exclude '.git' --exclude '__pycache__' -e "${e}" "${LOCAL_REPO}/" "h100-103:${dest}/repo/tau-0-wm/"
}

tar_pipe() {
  local src="$1" dest_host_cmd="$2"
  # dest_host_cmd is unused; callers pass ssh argv after --
  :
}

push_a3() {
  local dest="/home/tau0wm"
  "${HW_SSH[@]}" atlas "mkdir -p ${dest}/code ${dest}/repo/tau-0-wm ${dest}/pretrained ${dest}/logs ${dest}/shim ${dest}/outputs ${dest}/hf"
  tar -C "${HERE}" --exclude '.git' --exclude '__pycache__' --exclude '*.jsonl' --exclude '*.log' -cf - . \
    | "${HW_SSH[@]}" atlas "tar -C ${dest}/code -xf -"
  tar -C "${LOCAL_REPO}" --exclude '.git' --exclude '__pycache__' -cf - . \
    | "${HW_SSH[@]}" atlas "tar -C ${dest}/repo/tau-0-wm -xf -"
  "${HW_SSH[@]}" atlas "cp -f ${dest}/code/huawei/sitecustomize.py ${dest}/shim/sitecustomize.py"
}

push_a2() {
  local dest="/home/tau0wm"
  "${A2_SSH[@]}" root@10.228.7.110 "mkdir -p ${dest}/code ${dest}/repo/tau-0-wm ${dest}/pretrained ${dest}/logs ${dest}/shim ${dest}/outputs ${dest}/hf"
  tar -C "${HERE}" --exclude '.git' --exclude '__pycache__' --exclude '*.jsonl' --exclude '*.log' -cf - . \
    | "${A2_SSH[@]}" root@10.228.7.110 "tar -C ${dest}/code -xf -"
  tar -C "${LOCAL_REPO}" --exclude '.git' --exclude '__pycache__' -cf - . \
    | "${A2_SSH[@]}" root@10.228.7.110 "tar -C ${dest}/repo/tau-0-wm -xf -"
  "${A2_SSH[@]}" root@10.228.7.110 "cp -f ${dest}/code/huawei/sitecustomize.py ${dest}/shim/sitecustomize.py"
}

start_js() {
  local host="$1" dest="$2"
  "${JS_SSH[@]}" "${host}" "chmod +x ${dest}/code/docker/*.sh
    bash ${dest}/code/docker/run_jiangsuan.sh
    if ! docker top tau0wm | grep -q setup_in_container.sh; then
      docker exec -d -e ROOT=/workspace -e SMOKE_DEV=0 -e CUDA_VISIBLE_DEVICES= tau0wm \
        bash /workspace/code/docker/setup_in_container.sh
    fi
    if [ ! -f ${dest}/pretrained/tau-0-wm/simulator/diffusion_pytorch_model.bin ]; then
      if [ ! -f ${dest}/logs/hf_download.pid ] || ! kill -0 \$(cat ${dest}/logs/hf_download.pid) 2>/dev/null; then
        setsid nohup env ROOT=${dest} bash ${dest}/code/docker/download_weights.sh > ${dest}/logs/hf_download.nohup 2>&1 &
        echo \$! > ${dest}/logs/hf_download.pid
      fi
    fi
    docker ps --filter name=^tau0wm$ --format '{{.Names}} {{.Status}}'"
}

start_cty() {
  local host="$1" dest="/data/extra/tau0wm"
  "${JS_SSH[@]}" "${host}" "chmod +x ${dest}/code/docker/*.sh
    bash ${dest}/code/docker/run_cty.sh
    if ! docker top tau0wm | grep -q setup_in_container.sh; then
      docker exec -d tau0wm bash -lc 'source /etc/profile.d/zz_ppucompat.sh; export ROOT=/workspace SMOKE_DEV=0 CUDA_VISIBLE_DEVICES=; bash /workspace/code/docker/setup_in_container.sh'
    fi
    if [ ! -f ${dest}/pretrained/tau-0-wm/simulator/diffusion_pytorch_model.bin ]; then
      if [ ! -f ${dest}/logs/hf_download.pid ] || ! kill -0 \$(cat ${dest}/logs/hf_download.pid) 2>/dev/null; then
        setsid nohup env ROOT=${dest} bash ${dest}/code/docker/download_weights.sh > ${dest}/logs/hf_download.nohup 2>&1 &
        echo \$! > ${dest}/logs/hf_download.pid
      fi
    fi
    docker ps --filter name=^tau0wm$ --format '{{.Names}} {{.Status}}'"
}

start_h100() {
  h100 "chmod +x /data1/tau0wm/code/docker/*.sh
    bash /data1/tau0wm/code/docker/run_h100.sh
    if ! sudo -n docker top tau0wm | grep -q setup_in_container.sh; then
      sudo -n docker exec -d -e ROOT=/workspace -e SMOKE_DEV=0 -e CUDA_VISIBLE_DEVICES= tau0wm \
        bash /workspace/code/docker/setup_in_container.sh
    fi
    sudo -n docker ps --filter name=^tau0wm$ --format '{{.Names}} {{.Status}}'"
}

start_a3() {
  local dest="/home/tau0wm"
  "${HW_SSH[@]}" atlas "chmod +x ${dest}/code/docker/*.sh ${dest}/code/huawei/run_docker.sh
    bash ${dest}/code/huawei/run_docker.sh
    if ! docker top tau0wm | grep -q setup_in_container.sh; then
      docker exec -d -e ROOT=${dest} -e TAU_ROOT=${dest} -e SMOKE_DEV=0 -e CUDA_VISIBLE_DEVICES= tau0wm \
        bash ${dest}/code/docker/setup_in_container.sh
    fi
    if [ ! -f ${dest}/pretrained/tau-0-wm/simulator/diffusion_pytorch_model.bin ]; then
      if [ ! -f ${dest}/logs/hf_download.pid ] || ! kill -0 \$(cat ${dest}/logs/hf_download.pid) 2>/dev/null; then
        setsid nohup env ROOT=${dest} bash ${dest}/code/docker/download_weights.sh > ${dest}/logs/hf_download.nohup 2>&1 &
        echo \$! > ${dest}/logs/hf_download.pid
      fi
    fi
    docker ps --filter name=^tau0wm$ --format '{{.Names}} {{.Status}}'"
}

start_a2() {
  local dest="/home/tau0wm"
  "${A2_SSH[@]}" root@10.228.7.110 "chmod +x ${dest}/code/docker/*.sh ${dest}/code/huawei/run_docker.sh
    bash ${dest}/code/huawei/run_docker.sh
    if ! docker top tau0wm | grep -q setup_in_container.sh; then
      docker exec -d -e ROOT=${dest} -e TAU_ROOT=${dest} -e SMOKE_DEV=0 -e CUDA_VISIBLE_DEVICES= tau0wm \
        bash ${dest}/code/docker/setup_in_container.sh
    fi
    if [ ! -f ${dest}/pretrained/tau-0-wm/simulator/diffusion_pytorch_model.bin ]; then
      if [ ! -f ${dest}/logs/hf_download.pid ] || ! kill -0 \$(cat ${dest}/logs/hf_download.pid) 2>/dev/null; then
        setsid nohup env ROOT=${dest} bash ${dest}/code/docker/download_weights.sh > ${dest}/logs/hf_download.nohup 2>&1 &
        echo \$! > ${dest}/logs/hf_download.pid
      fi
    fi
    docker ps --filter name=^tau0wm$ --format '{{.Names}} {{.Status}}'"
}

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(16b 16c cty-18 cty-19 h100 a3 a2)
fi

for t in "${TARGETS[@]}"; do
  case "${t}" in
    16b) push_js jiangsuan-16b /nvme2n1/tau0wm; start_js jiangsuan-16b /nvme2n1/tau0wm ;;
    16c) push_js jiangsuan-16c /nvme2n1/tau0wm; start_js jiangsuan-16c /nvme2n1/tau0wm ;;
    cty-18) push_js cty-18 /data/extra/tau0wm; start_cty cty-18 ;;
    cty-19) push_js cty-19 /data/extra/tau0wm; start_cty cty-19 ;;
    h100) push_h100; start_h100 ;;
    a3) push_a3; start_a3 ;;
    a2) push_a2; start_a2 ;;
    *) echo "[ERR] unknown target ${t}" >&2; exit 1 ;;
  esac
done

echo "[OK] containers launched; setup + weight download continue in background"
