#!/bin/bash
# 在华为 cosmos3 容器内执行。ROOT=/home/cosmos3。
set -euo pipefail
export ROOT="${COSMOS3_ROOT:-/home/cosmos3}"
export SRC="${COSMOS3_SRC:-${ROOT}/src/cosmos-framework}"
export CODE="${ROOT}/code"
export VENV="${ROOT}/venv"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
mkdir -p "${ROOT}/shim"
cp -f "${ROOT}/code/huawei/sitecustomize.py" "${ROOT}/shim/sitecustomize.py"
export PYTHONPATH="${SRC}:${ROOT}/shim:${PYTHONPATH:-}"
bash "${ROOT}/code/env/setup_ppu.sh"
