#!/usr/bin/env bash
# 把 UNITREE G1 SONIC 代码+预训练权重推到松龄 4090。
# 不拷本机 Blackwell conda/.venv（sm 对不上）。Isaac Sim 要在 4090 上另装。
# 只用 songling-214；57 不通就跳过。
set -euo pipefail
AGX_PASS="${AGX_PASS:-$(awk -F= '/^AGX_PASS=/{print substr($0,index($0,"=")+1)}' /home/xk/.config/tongbot/fleet.env)}"
RSH="sshpass -p ${AGX_PASS} ssh -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=15"
HOST="${HOST:-agilex@10.151.46.214}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/agilex/unitree_g1}"

echo "[deploy] ${HOST}:${REMOTE_ROOT}"
$RSH "${HOST}" "mkdir -p ${REMOTE_ROOT}/{data_Driven,VLA,datasets}"

rsync -a --info=progress2 -e "${RSH}" \
    --exclude '.git/' \
    --exclude 'annotate/outputs/' \
    --exclude 'experiments/ledger/' \
    --exclude 'docs/exoskeleton/' \
    --exclude '__pycache__/' \
    /home/xk/data_Driven/ "${HOST}:${REMOTE_ROOT}/data_Driven/"

rsync -a --info=progress2 -e "${RSH}" \
    --exclude '.git/' \
    --exclude '.conda/' \
    --exclude '.venv/' \
    --exclude 'runnings/' \
    --exclude 'wheels/' \
    --exclude '__pycache__/' \
    /home/xk/VLA/Isaac-GR00T/ "${HOST}:${REMOTE_ROOT}/VLA/Isaac-GR00T/"

rsync -a --info=progress2 -e "${RSH}" \
    --exclude '.git/' \
    --exclude 'motionbricks/out/' \
    /home/xk/VLA/IsaacLab/ "${HOST}:${REMOTE_ROOT}/VLA/IsaacLab/"

rsync -a --info=progress2 -e "${RSH}" \
    --exclude '.git/' \
    --exclude 'motionbricks/out/' \
    --exclude '.venv_data_collection/' \
    /home/xk/VLA/GR00T-WholeBodyControl/ "${HOST}:${REMOTE_ROOT}/VLA/GR00T-WholeBodyControl/"

if [ -d /home/xk/VLA/datasets/g1_cup_ik_sonic ]; then
    rsync -a --info=progress2 -e "${RSH}" \
        /home/xk/VLA/datasets/g1_cup_ik_sonic/ \
        "${HOST}:${REMOTE_ROOT}/datasets/g1_cup_ik_sonic/"
fi

echo "[deploy] symlink ~/VLA -> ${REMOTE_ROOT}（env.sh 默认找 ~/VLA）"
$RSH "${HOST}" "mkdir -p ${HOME}/VLA ${HOME}/VLA/datasets && \
  ln -sfn ${REMOTE_ROOT}/VLA/Isaac-GR00T ${HOME}/VLA/Isaac-GR00T && \
  ln -sfn ${REMOTE_ROOT}/VLA/IsaacLab ${HOME}/VLA/IsaacLab && \
  ln -sfn ${REMOTE_ROOT}/VLA/GR00T-WholeBodyControl ${HOME}/VLA/GR00T-WholeBodyControl && \
  ln -sfn ${REMOTE_ROOT}/datasets/g1_cup_ik_sonic ${HOME}/VLA/datasets/g1_cup_ik_sonic && \
  ln -sfn ${REMOTE_ROOT}/data_Driven ${HOME}/data_Driven"

echo "[deploy] done. 下一步在 214 上跑 bootstrap（不要拷本机 .conda）："
echo "  bash ${REMOTE_ROOT}/data_Driven/recipes/g1_cup_ik/bootstrap_4090.sh"
echo "  训：bash ${REMOTE_ROOT}/data_Driven/recipes/gr00t_sonic/train_4090.sh --dataset-path ~/VLA/datasets/g1_cup_ik_sonic"
