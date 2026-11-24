#!/usr/bin/env bash
# 从 h100-103 只拉 τ0-WM 00036 EE 推理权重：step_N 的 transformer bin（约 11G）。
# 不要拉 latest_accelerator_state（optimizer / ZeRO）。
# 权重之后再拉 run 根 train.log / tfevents（后来加的 debug 不在 step_N）。
#
#   bash recipes/tau0_wm/pull_tau_ckpt.sh
#   TAU_CKPT_STEP=5000 bash recipes/tau0_wm/pull_tau_ckpt.sh
set -euo pipefail

SSH='ssh -F /home/xk/VLA/ssh_config -o ControlPath=/tmp/sshcm/%r@%h:%p'
REMOTE=h100-103
REMOTE_ROOT="${TAU_H100_OUTPUT:-/data1/tau0wm/outputs/tongbot_00036_ee/tau0wm_vam5b_ee_rela_192x256_bs1}"
DST="${TAU_CKPT:-/home/xk/VLA/h100_ckpts/tau0wm_tongbot_00036_ee}"

/home/xk/bin/h100 true >/dev/null

if [[ -n "${TAU_CKPT_STEP:-}" ]]; then
  STEP_DIR="${REMOTE_ROOT}/step_${TAU_CKPT_STEP}"
else
  STEP_DIR="$($SSH "$REMOTE" "ls -d ${REMOTE_ROOT}/step_* 2>/dev/null | awk -F_ '{print \$NF\" \"\$0}' | sort -n | tail -1 | cut -d' ' -f2-")"
fi

if [[ -z "${STEP_DIR}" ]]; then
  echo "远端还没有 step_* ckpt: ${REMOTE_ROOT}" >&2
  echo "训练 steps_to_save=5000，终档 step_30000。可先 --pretrained-only 冒烟。" >&2
  exit 1
fi

echo "remote $STEP_DIR -> $DST"
mkdir -p "$DST"
# 只要推理权重，排除 accelerator / optimizer
rsync -a --partial --info=progress2 -e "$SSH" \
  --exclude='latest_accelerator_state/' \
  --exclude='optimizer/' \
  --exclude='*.optimizer/' \
  "${REMOTE}:${STEP_DIR}/" "$DST/"
# run 根 train log / TB（debug 不在 step_N；缺文件 tar 为空即可）
$SSH "$REMOTE" "cd '${REMOTE_ROOT}' && tar cf - train.log events.out.tfevents* *.hw.jsonl 2>/dev/null" \
  | tar xf - -C "$DST" || true
echo "DONE"
ls -lh "$DST" | head -20
