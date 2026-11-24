#!/bin/bash
# 江算 cosmos3 容器内 source。
ROOT="${ROOT:-/workspace}"
set +u
source /usr/local/PPU_SDK/envsetup.sh
source "${ROOT}/venv/bin/activate"
set +u
export ROOT
export SRC="${SRC:-${ROOT}/src/cosmos-framework}"
export PYTHONPATH="${SRC}:${ROOT}/shim:${PYTHONPATH:-}"
export HF_HOME="${ROOT}/hf"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export I4_ATTN_BACKENDS=sdpa
export TORCHDYNAMO_DISABLE=1
export WALLX_SKIP_CUDA_EXT=1
export PYTHONUNBUFFERED=1
export COSMOS3_MODEL="${COSMOS3_MODEL:-${ROOT}/pretrained/Cosmos3-Nano}"
export COSMOS3_FRAMEWORK="${COSMOS3_FRAMEWORK:-${SRC}}"
echo "python=$(which python) ROOT=${ROOT} MODEL=${COSMOS3_MODEL}"
