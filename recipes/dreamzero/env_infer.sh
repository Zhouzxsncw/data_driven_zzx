#!/usr/bin/env bash
# DreamZero local infer env (RTX PRO 6000 Blackwell).
# Usage: source /home/xk/VLA/dreamzero_staging/env_infer.sh
export DREAMZERO_PATH="${DREAMZERO_PATH:-/home/xk/VLA/dreamzero}"
export EMBODIED_PATH="${EMBODIED_PATH:-/home/xk/VLA/RLinf/examples/sft}"
export PYTHONPATH="${DREAMZERO_PATH}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
# Prebuilt flash-attn needs newer CXXABI than Ubuntu 22.04 system libstdc++.
export LD_LIBRARY_PATH="/home/xk/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Do not load torchcodec; it aborted on H100 and is unused (pyav).
export DISABLE_TORCHCODEC=1
source /data/dreamzero-infer/bin/activate
echo "python=$(which python) $(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
