#!/bin/bash
ROOT="${ROOT:-/workspace}"
set +u
source /usr/local/PPU_SDK/envsetup.sh
source "${ROOT}/venv/bin/activate"
set +u
export ROOT
export SRC="${SRC:-${ROOT}/src/Isaac-GR00T}"
export PYTHONPATH="${SRC}:${ROOT}/shim:${PYTHONPATH:-}"
export HF_HOME="${ROOT}/hf"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export GR00T_DIR="${GR00T_DIR:-${SRC}}"
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${ROOT}/pretrained/Cosmos-Reason2-2B}"
export GR00T_CKPT="${GR00T_CKPT:-${ROOT}/pretrained/GR00T-N1.7-3B}"
echo "python=$(which python) ROOT=${ROOT} CKPT=${GR00T_CKPT}"
