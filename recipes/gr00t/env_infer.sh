# GR00T N1.7 服务端环境
# source recipes/gr00t/env_infer.sh
#
# 环境优先级：conda gr00t_n17 → Isaac-GR00T/.conda
unalias -a 2>/dev/null || true

resolve_gr00t_python() {
    # 1) conda gr00t_n17（推理电脑）
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ] \
        && [ -d "${HOME}/miniconda3/envs/gr00t_n17" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        set +u
        conda activate gr00t_n17
        set +u
        echo "python"
        return
    fi
    if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ] \
        && [ -d "${HOME}/anaconda3/envs/gr00t_n17" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
        set +u
        conda activate gr00t_n17
        set +u
        echo "python"
        return
    fi
    # 2) Isaac-GR00T/.conda（GPU 训练节点）
    local gr00t_dir="${GR00T_DIR:-${HOME}/VLA/Isaac-GR00T}"
    if [ -x "${gr00t_dir}/.conda/bin/python" ]; then
        echo "${gr00t_dir}/.conda/bin/python"
        return
    fi
    echo ""
}

GR00T_PYTHON="$(resolve_gr00t_python)"
if [ -z "${GR00T_PYTHON}" ]; then
    echo "[env_infer] ERROR: 找不到 gr00t_n17 conda 或 Isaac-GR00T/.conda/bin/python" >&2
    return 1 2>/dev/null || exit 1
fi

export GR00T_DIR="${GR00T_DIR:-${HOME}/VLA/Isaac-GR00T}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# Cosmos backbone 自动检测
if [ -z "${GR00T_BACKBONE_PATH:-}" ]; then
    for bb in \
        "${GR00T_DIR}/nvidia/Cosmos-Reason2-2B" \
        "${GR00T_DIR}/checkpoints/Cosmos-Reason2-2B" \
        "${HOME}/Isaac-GR00T/nvidia/Cosmos-Reason2-2B"; do
        if [ -d "${bb}" ]; then
            GR00T_BACKBONE_PATH="${bb}"
            break
        fi
    done
fi
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-}"

echo "python=$(${GR00T_PYTHON} --version 2>&1)"
${GR00T_PYTHON} -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')" 2>/dev/null || true
