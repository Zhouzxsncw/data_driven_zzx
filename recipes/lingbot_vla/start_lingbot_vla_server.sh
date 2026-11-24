#!/usr/bin/env bash
# ============================================================
# LingBot-VLA V2 WebSocket 推理服务 — 对齐推理电脑 cigai20
# ============================================================
# 当前默认：00086 global_step 9287（jiangsuan-4 最终 ckpt）
#   本机: /home/xk/VLA/h100_ckpts/00086_lingbotvla/checkpoint/9287/hf_ckpt
#
# 用法:
#   bash recipes/lingbot_vla/start_lingbot_vla_server.sh
#   bash recipes/lingbot_vla/start_lingbot_vla_server.sh [MODEL_PATH] [PORT]
#
# 环境：本机 conda lingbotvla（torch 2.8 + transformers 4.57 + flash_attn）
# 客户端另开: bash recipes/lingbot_vla/start_lingbot_vla_batch_infer.sh
# ============================================================
set -euo pipefail

REPO="${LINGBOT_VLA_REPO:-/home/xk/VLA/lingbot-vla-v2}"
PYTHON="${LINGBOTVLA_PYTHON:-/home/xk/miniconda3/envs/lingbotvla/bin/python}"
MODEL_PATH="${1:-/home/xk/VLA/h100_ckpts/00086_lingbotvla/checkpoint/9287/hf_ckpt}"
PORT="${2:-8006}"
QWEN_DEFAULT="/home/xk/VLA/lingbot-vla-v2/pretrained/qwen3-tokenizer"
LOG="${LINGBOT_VLA_SERVER_LOG:-/home/xk/VLA/h100_ckpts/00086_lingbotvla/server_9287.log}"

export QWEN3VL_PATH="${QWEN3VL_PATH:-${QWEN_DEFAULT}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -x "${PYTHON}" ]; then
    echo "[ERROR] 找不到 lingbotvla python: ${PYTHON}" >&2
    exit 1
fi
if [ ! -f "${REPO}/deploy/lingbot_vla_v2_policy.py" ]; then
    echo "[ERROR] 找不到 ${REPO}/deploy/lingbot_vla_v2_policy.py" >&2
    exit 1
fi
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[ERROR] 权重未就绪: ${MODEL_PATH}" >&2
    echo "  需要含 config.json 与 model-*-of-00006.safetensors 的 hf_ckpt。" >&2
    exit 1
fi
if [ ! -f "${MODEL_PATH}/../.."/lingbotvla_cli.yaml ] && [ ! -f "${MODEL_PATH}/../../../lingbotvla_cli.yaml" ]; then
    echo "[WARN] 未在 hf_ckpt 上三级目录找到 lingbotvla_cli.yaml，deploy 可能读不到包内 norm_stats"
fi
if [ ! -d "${QWEN3VL_PATH}" ]; then
    echo "[ERROR] Qwen3-VL tokenizer 路径无效: ${QWEN3VL_PATH}" >&2
    exit 1
fi

CLI_YAML="$(python3 - << PY
from pathlib import Path
p = Path("${MODEL_PATH}").resolve()
for _ in range(4):
    cand = p / "lingbotvla_cli.yaml"
    if cand.is_file():
        print(cand)
        break
    p = p.parent
PY
)"

echo "============================================================"
echo "  LingBot-VLA V2 推理服务（00086 / step 9287）"
echo "============================================================"
echo "  Python:    ${PYTHON} ($("${PYTHON}" --version 2>&1))"
echo "  仓库:      ${REPO}"
echo "  模型:      ${MODEL_PATH}"
echo "  Qwen3VL:   ${QWEN3VL_PATH}"
echo "  cli yaml:  ${CLI_YAML:-（未找到）}"
echo "  端口:      ${PORT}"
echo "  日志:      ${LOG}"
echo "============================================================"

mkdir -p "$(dirname "${LOG}")"
cd "${REPO}"

OLD_PIDS=$(pgrep -f "deploy.lingbot_vla_v2_policy" || true)
if [ -n "${OLD_PIDS}" ]; then
    echo "杀掉旧 LingBot-VLA 服务: ${OLD_PIDS}"
    echo "${OLD_PIDS}" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

{
    echo ""
    echo "===== $(date -Iseconds) server restart model=${MODEL_PATH} port=${PORT} ====="
} >> "${LOG}"

echo "前台启动。加载约 1–2 分钟，出现 Model initialized / Connection from 后再开客户端。"
echo "Ctrl+C 停止服务。"
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../common" && pwd)/start_lower_machine.sh" || \
  echo "[WARN] 下位机未拉起。SKIP_LOWER_MACHINE=1 可跳过。" >&2
exec "${PYTHON}" -m deploy.lingbot_vla_v2_policy \
    --model_path "${MODEL_PATH}" \
    --use_length 50 \
    --chunk_ret True \
    --use_bf16 True \
    --use_compile False \
    --port "${PORT}" \
    > >(tee -a "${LOG}") 2>&1
