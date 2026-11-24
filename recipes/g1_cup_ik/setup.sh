#!/usr/bin/env bash
# Isaac Sim 5.1 已 pip 安装后：把 Isaac Lab 扩展装进 env_isaaclab。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
activate_isaaclab

if ! python -c "import isaacsim" 2>/dev/null; then
    echo "[g1_cup_ik] 先装 isaacsim：pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com" >&2
    exit 1
fi

cd "${ISAACLAB_DIR}"
./isaaclab.sh --install none
python -m pip install -q pandas pyarrow
echo "[g1_cup_ik] Isaac Lab 扩展已装进 ${ISAACLAB_CONDA}（含 pandas/pyarrow 写 LeRobot）"
