#!/usr/bin/env bash
# 从 h100-103 只拉 GWP-0.5 00073 推理所需：transformer/（约 12G）+ norm_stats。
# 不要拉 checkpoint 里 79G 的 pytorch_model（optimizer / ZeRO 分片）。
#
#   bash recipes/gigaworld_policy/pull_gwp_ckpt.sh
set -euo pipefail

SSH='ssh -F /home/xk/VLA/ssh_config -o ControlPath=/tmp/sshcm/%r@%h:%p'
REMOTE=h100-103
SRC=/data1/gwp/outputs/00073_gwp05_ft/models/checkpoint_epoch_22_step_50000/transformer/
DST="${GWP_CKPT:-/home/xk/VLA/h100_ckpts/gwp05_tongbot_00073}/transformer/"
NORM_SRC=/data1/gwp/norm_stats_00073.json
NORM_DST="${GWP_CKPT:-/home/xk/VLA/h100_ckpts/gwp05_tongbot_00073}/norm_stats_00073.json"

# 确保 ControlMaster 活着
/home/xk/bin/h100 true >/dev/null

mkdir -p "$(dirname "$DST")" "$(dirname "$NORM_DST")"
echo "norm  $NORM_SRC -> $NORM_DST"
rsync -a --partial -e "$SSH" "${REMOTE}:${NORM_SRC}" "$NORM_DST"
echo "transformer  $SRC -> $DST"
echo "约 12G，网关慢时可能要数小时；可重跑续传。"
rsync -a --partial --info=progress2 -e "$SSH" "${REMOTE}:${SRC}" "$DST"
echo "DONE"
ls -lh "$DST" "$NORM_DST"
