#!/bin/bash
set -euo pipefail
export ROOT="${GR00T_ROOT:-/home/gr00t}"
export SRC="${GR00T_SRC:-${ROOT}/src/Isaac-GR00T}"
export VENV="${ROOT}/venv"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
mkdir -p "${ROOT}/shim"
cp -f "${ROOT}/code/huawei/sitecustomize.py" "${ROOT}/shim/sitecustomize.py"
export PYTHONPATH="${SRC}:${ROOT}/shim:${PYTHONPATH:-}"
bash "${ROOT}/code/env/setup_ppu.sh"
