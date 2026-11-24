#!/bin/bash
# ============================================================
# GR00T N1.7 推理服务器 — 对齐推理电脑 cigai20 / infer-pc
# ============================================================
# 当前默认：gpu-node1 00073 full_allow_padding checkpoint-41000
#   embodiment     NEW_EMBODIMENT
#   模态配置       Isaac-GR00T/examples/00073_tong_dual_arm/00073_config.py
#   相机           observation.images.{left,front,right}
#   state/action   observation.state 16D RELATIVE
#   language       annotation.human.task_description
#
# 用法:
#   ./start_gr00t_server.sh [MODEL_PATH] [EMBODIMENT_TAG] [PORT]
#
# 环境：本机用 Isaac-GR00T/.conda；推理电脑用 conda gr00t_n17。
# 官方基座 GR00T-N1.7-3B 不含 new_embodiment，必须用微调 checkpoint。
# ============================================================

set -euo pipefail

resolve_gr00t_dir() {
    if [ -n "${GR00T_DIR:-}" ] && [ -d "${GR00T_DIR}" ]; then
        echo "${GR00T_DIR}"
        return
    fi
    if [ -d "${HOME}/VLA/Isaac-GR00T" ]; then
        echo "${HOME}/VLA/Isaac-GR00T"
        return
    fi
    if [ -d "${HOME}/Isaac-GR00T" ]; then
        echo "${HOME}/Isaac-GR00T"
        return
    fi
    echo ""
}

has_finetune_ckpt() {
    local d="$1"
    [ -d "${d}" ] || return 1
    [ -f "${d}/config.json" ] || return 1
    if [ -f "${d}/processor_config.json" ] || [ -d "${d}/processor" ]; then
        return 0
    fi
    # 训练中途目录可能只有 experiment_cfg，不能当推理权重
    return 1
}

GR00T_DIR="$(resolve_gr00t_dir)"
if [ -z "${GR00T_DIR}" ]; then
    echo "[ERROR] 找不到 Isaac-GR00T（试过 ~/VLA/Isaac-GR00T 与 ~/Isaac-GR00T）"
    exit 1
fi

# ---- python：推理机 conda gr00t_n17；GPU 节点可用 Isaac-GR00T/.conda ----
if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ] \
    && [ -d "${HOME}/anaconda3/envs/gr00t_n17" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate gr00t_n17
    PYTHON="python"
elif [ -x "${GR00T_DIR}/.conda/bin/python" ]; then
    PYTHON="${GR00T_DIR}/.conda/bin/python"
else
    echo "[ERROR] 找不到 gr00t_n17 conda 或 ${GR00T_DIR}/.conda/bin/python"
    exit 1
fi

# Cosmos backbone：与推理电脑一样用本地替身（Qwen3-VL 架构，权重由 ckpt 覆盖）
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
if [ -z "${GR00T_BACKBONE_PATH:-}" ]; then
    for bb in \
        "${GR00T_DIR}/nvidia/Cosmos-Reason2-2B" \
        "${GR00T_DIR}/checkpoints/Cosmos-Reason2-2B" \
        "${HOME}/Isaac-GR00T/nvidia/Cosmos-Reason2-2B"
    do
        if [ -d "${bb}" ]; then
            GR00T_BACKBONE_PATH="${bb}"
            break
        fi
    done
fi
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-}"

EMBODIMENT_TAG="${2:-NEW_EMBODIMENT}"
PORT="${3:-5555}"

if [ -n "${1:-}" ]; then
    MODEL_PATH="$1"
else
    MODEL_PATH=""
    for cand in \
        "${HOME}/VLA/gr00t_n17_outputs/finetune_00073_full_allow_padding/checkpoint-41000" \
        "${HOME}/gr00t_models/finetune_00073" \
        "${HOME}/VLA/gr00t_n17_outputs/finetune_00036"
    do
        if has_finetune_ckpt "${cand}"; then
            MODEL_PATH="${cand}"
            break
        fi
    done
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] 未找到微调 checkpoint。"
    echo "  需要含 config.json 以及 processor/ 或 processor_config.json 的目录。"
    echo "  用法: $0 /path/to/checkpoint-41000 [NEW_EMBODIMENT] [5555]"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[ERROR] 模型路径无效: ${MODEL_PATH}"
    exit 1
fi

if ! has_finetune_ckpt "${MODEL_PATH}"; then
    echo "[ERROR] ${MODEL_PATH} 不像微调 ckpt（缺 processor）。"
    echo "  官方基座不能用 --embodiment-tag NEW_EMBODIMENT。"
    exit 1
fi

echo "============================================================"
echo "  GR00T N1.7 推理服务器（NEW_EMBODIMENT）"
echo "============================================================"
echo "  Isaac-GR00T:   ${GR00T_DIR}"
echo "  模型:          ${MODEL_PATH}"
echo "  Backbone:      ${GR00T_BACKBONE_PATH}"
echo "  Embodiment:    ${EMBODIMENT_TAG}"
echo "  端口:          ${PORT}"
echo "  Python:        $(${PYTHON} --version 2>&1)"
echo "============================================================"

cd "${GR00T_DIR}"

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../common" && pwd)/start_lower_machine.sh" || \
  echo "[WARN] 下位机未拉起。SKIP_LOWER_MACHINE=1 可跳过。" >&2

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" gr00t/eval/run_gr00t_server.py \
    --model-path "${MODEL_PATH}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --port "${PORT}" \
    --device cuda
