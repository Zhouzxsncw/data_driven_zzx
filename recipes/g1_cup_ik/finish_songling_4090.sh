#!/usr/bin/env bash
# 等 deploy_songling_4090.sh 结束后：补拷脚本/数据，在 214 上 bootstrap 训练环境。
set -euo pipefail
DEPLOY_PID="${DEPLOY_PID:-1539553}"
LOG="${LOG:-/home/xk/VLA/gr00t_sonic_outputs/g1_cup_ik_sonic/deploy_songling_4090.log}"
AGX_PASS="${AGX_PASS:-$(awk -F= '/^AGX_PASS=/{print substr($0,index($0,"=")+1)}' /home/xk/.config/tongbot/fleet.env)}"
RSH="sshpass -p ${AGX_PASS} ssh -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=20"
HOST="agilex@10.151.46.214"

echo "[finish] wait deploy pid=${DEPLOY_PID} $(date '+%F %T')" | tee -a "${LOG}"
while kill -0 "${DEPLOY_PID}" 2>/dev/null; do
    sleep 30
done
echo "[finish] deploy exited $(date '+%F %T')" | tee -a "${LOG}"

# 再跑一遍 deploy 尾部：IsaacLab / WBC / dataset（若主脚本已做完会很快）
bash /home/xk/data_Driven/recipes/g1_cup_ik/deploy_songling_4090.sh >> "${LOG}" 2>&1 || true

if [ -d /home/xk/VLA/datasets/g1_cup_ik_sonic_full ]; then
    rsync -a --info=progress2 -e "${RSH}" \
        /home/xk/VLA/datasets/g1_cup_ik_sonic_full/ \
        "${HOST}:/home/agilex/unitree_g1/datasets/g1_cup_ik_sonic_full/"
    $RSH "${HOST}" "ln -sfn /home/agilex/unitree_g1/datasets/g1_cup_ik_sonic_full /home/agilex/VLA/datasets/g1_cup_ik_sonic_full"
fi

echo "[finish] bootstrap_4090 on 214 $(date '+%F %T')" | tee -a "${LOG}"
$RSH "${HOST}" 'bash /home/agilex/unitree_g1/data_Driven/recipes/g1_cup_ik/bootstrap_4090.sh' \
    >> /home/xk/VLA/gr00t_sonic_outputs/g1_cup_ik_sonic/bootstrap_4090.log 2>&1
echo "[finish] bootstrap exit=$? $(date '+%F %T')" | tee -a "${LOG}"
