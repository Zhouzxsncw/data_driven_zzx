#!/usr/bin/env bash
# τ0-WM @ h100-103 的环境（uv venv 版，对应 gpu-master 侧的 env.sh / conda 版）。
#
#   source /data1/tau0wm/local_scripts/h100/env_h100.sh
#   bash /data1/tau0wm/local_scripts/launch_train.sh --smoke
#
# launch_train.sh 里的 TAU_ROOT / TAU_PY / TAU_ENV_SH / TAU_RUNS 全部由这里给出，
# 所以**必须先 source 再跑 launch**（launch 内部还会再 source 一次，幂等）。
# 这个文件只允许做export 与字符串处理，不能有非零返回的命令：
# launch_train.sh 带着 `set -euo pipefail` source 它，任何命令失败都会终止启动。
unalias -a 2>/dev/null || true

TAU_H100_ROOT="${TAU_H100_ROOT:-/data1/tau0wm}"
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TAU_ROOT="${TAU_ROOT:-${TAU_H100_ROOT}/repo/tau-0-wm}"
export TAU_RECIPE="${TAU_H100_ROOT}/local_scripts"
export TAU_WEIGHTS="${TAU_H100_ROOT}/pretrained/tau-0-wm"
export WAN_ROOT="${WAN_ROOT:-${TAU_H100_ROOT}/pretrained/Wan2.2-TI2V-5B}"
export TAU_RUNS="${TAU_RUNS:-${TAU_H100_ROOT}/outputs}"
export TAU_LOGS="${TAU_LOGS:-${TAU_H100_ROOT}/logs}"

export VIRTUAL_ENV="${TAU_H100_ROOT}/envs/tau0wm"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export TAU_PY="${VIRTUAL_ENV}/bin/python"
export TAU_ENV_SH="${_HERE}/env_h100.sh"
export CONFIG="${CONFIG:-${TAU_RECIPE}/configs/tau_model/posttrain_tongbot_00036_ee.yaml}"

export PYTHONPATH="${TAU_ROOT}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1          # hf-mirror 不代理 xet CAS
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="9.0"    # H100 = sm90
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_MODE=offline

echo "tau0wm(h100): py=$(command -v python) TAU_ROOT=${TAU_ROOT} runs=${TAU_RUNS}"
