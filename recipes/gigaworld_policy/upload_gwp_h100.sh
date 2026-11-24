#!/usr/bin/env bash
# 把 GigaWorld-Policy 训练栈(代码/env/权重/v3.0数据)串行上传到 h100-103:/data1/gwp
set -uo pipefail
SSH="ssh -F /home/xk/VLA/ssh_config -o ControlPath=/tmp/sshcm/%r@%h:%p"
RS="rsync -a --partial --whole-file -e \"$SSH\""
LOG=/home/xk/data_Driven/recipes/gigaworld_policy/upload_gwp_h100.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
REMOTE=h100-103

$SSH $REMOTE 'mkdir -p /data1/gwp/envs /data1/gwp/pretrained /data1/dataset' || exit 1

log "1/5 代码 259M"
rsync -a --delete -e "$SSH" --exclude '__pycache__' --exclude 'pretrained' \
  /home/xk/VLA/giga-world-policy/ $REMOTE:/data1/gwp/giga-world-policy/ || { log "FATAL code"; exit 2; }

log "2/5 conda env 打包(本地)"
ENVDIR=/home/xk/miniconda3/envs/gigaworld-policy
tar -cf /tmp/gwp-env.tar -C $(dirname $ENVDIR) $(basename $ENVDIR) || { log "FATAL tar"; exit 3; }
ls -la /tmp/gwp-env.tar
log "2/5 env 上传"
rsync -a --partial -e "$SSH" /tmp/gwp-env.tar $REMOTE:/data1/gwp/envs/ || { log "FATAL env rsync"; exit 4; }
$SSH $REMOTE 'cd /data1/gwp/envs && tar -xf gwp-env.tar && rm gwp-env.tar && ls envs 2>/dev/null; /data1/gwp/envs/gigaworld-policy/bin/python --version' || { log "FATAL env 解包"; exit 5; }

log "3/5 数据集 v3.0 8.8G"
rsync -a --partial -e "$SSH" /data/wall-oss/datasets/00073_rm75_ee_v30/ $REMOTE:/data1/dataset/00073_rm75_ee_v30/ || { log "FATAL data"; exit 6; }

log "4/5 Giga-World-Policy-0.5 权重 23G"
rsync -a --partial -e "$SSH" /data/giga-world-policy/pretrained/Giga-World-Policy-0.5/ $REMOTE:/data1/gwp/pretrained/Giga-World-Policy-0.5/ || { log "FATAL gwp wt"; exit 7; }

log "5/5 Wan2.2-TI2V-5B-Diffusers 32G"
rsync -a --partial -e "$SSH" /data/giga-world-policy/pretrained/Wan2.2-TI2V-5B-Diffusers/ $REMOTE:/data1/gwp/pretrained/Wan2.2-TI2V-5B-Diffusers/ || { log "FATAL wan"; exit 8; }

log "SUCCESS 全部上传完成,校验:"
$SSH $REMOTE 'du -sh /data1/gwp/* /data1/dataset/00073_rm75_ee_v30; /data1/gwp/envs/gigaworld-policy/bin/python -c "import torch;print(torch.__version__)"'
