#!/usr/bin/env bash
# OpenPI π0-FAST 本机服务端（端口 1243，JAX venv，与 00036 训练共卡 ~20GB）
#
#   bash recipes/pi0fast/launch_pi0fast_server.sh
#   bash recipes/pi0fast/launch_pi0fast_server.sh --load-only
set -euo pipefail

STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$STAGING/../.." && pwd)"
INFER_ROOT="${INFER_ROOT:-${REPO_ROOT}/infer}"
cd "$STAGING"

PYTHON="${PI0FAST_PYTHON:-/home/xk/VLA/openpi/cigai-agilex-train-openpi/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] 找不到 Python: $PYTHON" >&2
    exit 1
fi

# 训练 run 根目录（policy_config 自动解析最新 step 的 pretrained_model）
CKPT="${PI0FAST_CKPT:-/home/xk/VLA/openpi/cigai-agilex-train-openpi/checkpoints/00036_pi0fast_bs16_lr2.5e-5_ah15_1gpu}"
PORT="${PI0FAST_PORT:-1243}"
LOG="${STAGING}/server_pi0fast_00036.log"
PROMPT="${PI0FAST_PROMPT:-With both hands, pick up the yellow banana and the yellow lemon and put them into the black box; then take the green vegetables and place them into the blue box.}"

if [ ! -f "${CKPT}/model.safetensors" ] && [ ! -d "${CKPT}/params" ] && [ ! -d "${CKPT}/checkpoints" ]; then
    echo "[ERROR] 权重未就绪: ${CKPT}" >&2
    echo "        训练首个 ckpt（step 2000）产出前请等待；或指定 PI0FAST_CKPT" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
# 训练已占 ~73GB；推理只分配剩余 ~20GB
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export TOKENIZERS_PARALLELISM=false

setsid nohup "$PYTHON" "$STAGING/serve_tong_pi0fast.py" \
    --ckpt "$CKPT" \
    --port "$PORT" \
    --instruction "$PROMPT" \
    --config-name pi0_fast_tongbot \
    --action-dim 16 \
    --action-horizon 15 \
    --delta-action-dims "7,7,-1,-1" \
    > "$LOG" 2>&1 &
echo "server pid=$! log=$LOG"
sleep 3
tail -3 "$LOG" || true
bash "$STAGING/../common/start_lower_machine.sh" || \
  echo "[WARN] 下位机未拉起。重试: bash $STAGING/../common/start_lower_machine.sh ；跳过: SKIP_LOWER_MACHINE=1" >&2
