# G1 Dex3 抓杯：Isaac Lab GPU 差分 IK 采集
set -euo pipefail

_RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RECIPE_DIR="${_RECIPE_DIR}"
export REPO_ROOT="$(cd "${_RECIPE_DIR}/../.." && pwd)"
if [ -z "${ISAACLAB_DIR:-}" ]; then
    if [ -d "${HOME}/VLA/IsaacLab" ]; then
        export ISAACLAB_DIR="${HOME}/VLA/IsaacLab"
    elif [ -d "${HOME}/unitree_g1/VLA/IsaacLab" ]; then
        export ISAACLAB_DIR="${HOME}/unitree_g1/VLA/IsaacLab"
    else
        export ISAACLAB_DIR="${HOME}/VLA/IsaacLab"
    fi
fi
export ISAACLAB_CONDA="${ISAACLAB_CONDA:-env_isaaclab}"
export G1_CUP_IK_OUT="${G1_CUP_IK_OUT:-${HOME}/VLA/datasets/g1_cup_ik}"
# 带 ego_view 的 LeRobot v2.1（给 recipes/gr00t_sonic/train.sh）
export G1_CUP_SONIC_OUT="${G1_CUP_SONIC_OUT:-${HOME}/VLA/datasets/g1_cup_ik_sonic}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
# nohup 会设 TERM=dumb；isaaclab.sh 的 `tabs 4` + set -e 会直接退出
export TERM=xterm-256color

# 松龄 1×4090 24GB：少 env / 低 batch。本机 96GB 不要走这条。
_g1_maybe_4090() {
    if [ "${UNITREE_4090:-0}" = "1" ]; then
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi '4090'
        return $?
    fi
    return 1
}
if _g1_maybe_4090 && [ -f "${_RECIPE_DIR}/env_4090.sh" ]; then
    # shellcheck disable=SC1091
    source "${_RECIPE_DIR}/env_4090.sh"
fi

activate_isaaclab() {
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${ISAACLAB_CONDA}"
    export OMNI_KIT_ACCEPT_EULA=YES
    # isaaclab.sh uses `tabs 4` + set -e; dumb TERM 会直接退出
    export TERM="${TERM:-xterm-256color}"
}

isaaclab_python() {
    if [ ! -x "${ISAACLAB_DIR}/isaaclab.sh" ]; then
        echo "[g1_cup_ik] 找不到 ${ISAACLAB_DIR}/isaaclab.sh" >&2
        return 1
    fi
    printf '%s\n' "${ISAACLAB_DIR}/isaaclab.sh"
}
