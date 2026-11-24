#!/usr/bin/env bash
# τ0-WM 训练环境（gpu-master / Blackwell sm_120）
#
# 与官方 requirements.txt 的差异（都是刻意取舍，勿"照抄修复"）：
#   torch/torchvision/torchaudio 2.7.1  →  2.7.1+cu128
#       官方锁的是 cu126 轮子，arch_list 不含 sm_120，Blackwell 上会 no kernel image available。
#   xformers / flash-attn               →  不装
#       models/.../transformers/attention.py 有 SDPA 自动回退，装了反而在 sm_120 上编译失败。
#   pin / libpinocchio / libcoal / eigenpy / coal / cmeel-*  →  不装
#       全仓库 grep 无任何 import，只服务 eef 运动学分支；我们走 joint 空间用不到。
#   deepspeed                           →  不装
#       配置里 use_deepspeed: false，DeepSpeedPlugin 只在分支内实例化，顶层 import 不需要它。
#       （将来上多卡 ZeRO-1 再补装。）
#   hf_transfer / hf-xet                →  不装
#       huggingface_hub 新版已弃用 hf_transfer；xet 走 hf-mirror 会 401。
#   msgpack-python                      →  不装
#       0.5.6 在 py3.11 构建必挂，requirements 里同时有 msgpack==1.1.2，前者是冗余项。
#   lerobot 0.4.0 → 0.4.4 --no-deps
#       0.4.4 是本机已验证能读 TongBot v3.0 + av1/pyav 的版本；--no-deps 防止它回退 torch。
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

CONDA_ENV="${CONDA_ENV:-tau0wm}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAU_ROOT="${TAU_ROOT:-/home/xk/VLA/tau-0-wm}"
LOG_FILE="${LOG_FILE:-${HERE}/setup_env.log}"

: > "${LOG_FILE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"; }

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  log "找不到 conda（~/miniconda3）" >&2; exit 1
fi

if ! conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
  log "创建 conda env ${CONDA_ENV}（python 3.11）"
  conda create -n "${CONDA_ENV}" python=3.11 -y >>"${LOG_FILE}" 2>&1
fi
# shellcheck disable=SC1091
conda activate "${CONDA_ENV}"
PY="$(which python)"
log "python = ${PY} $(${PY} -V 2>&1)"

log "[1/4] torch 2.7.1+cu128（Blackwell 必须 cu128）"
"${PY}" -m pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128 >>"${LOG_FILE}" 2>&1

log "[2/4] 训练主依赖（版本按官方 requirements 锁，torch 系除外）"
"${PY}" -m pip install --no-cache-dir \
    "accelerate==1.12.0" \
    "diffusers==0.35.2" \
    "transformers==4.57.6" \
    "tokenizers==0.22.2" \
    "huggingface-hub==0.35.3" \
    "safetensors==0.6.2" \
    "datasets==4.1.1" \
    "pyarrow==23.0.1" \
    "pandas==2.3.1" \
    "numpy==2.2.6" \
    "einops==0.8.1" \
    "omegaconf==2.3.0" \
    "PyYAML==6.0.2" \
    "pyyaml-include==1.4.1" \
    "av==15.1.0" \
    "kornia==0.8.2" \
    "tensorboard==2.20.0" \
    "tqdm==4.67.1" \
    "scipy==1.15.3" \
    "matplotlib==3.10.5" \
    "moviepy==1.0.3" \
    "imageio==2.37.2" \
    "imageio-ffmpeg==0.6.0" \
    "msgpack==1.1.2" \
    "websockets==14.1" \
    "sentencepiece==0.2.0" \
    "ftfy==6.3.1" \
    "pillow==11.3.0" \
    "opencv-python-headless==4.10.0.84" \
    "filelock==3.20.1" \
    "requests==2.32.5" \
    "psutil==7.1.3" \
    "packaging" \
    "termcolor" \
    "jsonlines" \
    "deepdiff" \
    "draccus==0.10.0" \
    "dm-tree" \
    "gymnasium" \
    >>"${LOG_FILE}" 2>&1 || log "WARN：主依赖有冲突，见日志，下一步手工处理"

log "[3/4] lerobot 0.4.4（--no-deps，保住刚装好的 cu128 torch）"
"${PY}" -m pip install --no-cache-dir --no-deps "lerobot==0.4.4" >>"${LOG_FILE}" 2>&1

log "[4/4] 复核 torch 仍是 cu128（被回退过就重敲一次）"
if ! "${PY}" -c "import torch,sys;sys.exit(0 if '+cu128' in torch.__version__ else 1)" 2>/dev/null; then
  log "torch 被回退了，重装 cu128"
  "${PY}" -m pip install --no-cache-dir --force-reinstall --no-deps \
      torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
      --index-url https://download.pytorch.org/whl/cu128 >>"${LOG_FILE}" 2>&1
fi

cd "${TAU_ROOT}"
log "=== 自检 ==="
"${PY}" - <<'PYEOF' 2>&1 | tee -a "${LOG_FILE}"
import sys, torch
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "avail", torch.cuda.is_available(),
      "sm", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
assert any('sm_120' in a or '12.0' in a for a in torch.cuda.get_arch_list()), torch.cuda.get_arch_list()
import lerobot, transformers, diffusers, accelerate, av, kornia, einops, omegaconf
print("lerobot", lerobot.__version__, "| transformers", transformers.__version__,
      "| diffusers", diffusers.__version__, "| accelerate", accelerate.__version__,
      "| av", av.__version__)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
for m in ("_ensure_hf_dataset_loaded", "_get_query_indices", "_query_hf_dataset",
          "_get_query_timestamps", "_query_videos"):
    assert hasattr(LeRobotDataset, m), m
print("lerobot 私有 API 齐备（τ0-WM _MyLeRobotDatasetv30 依赖）")
import matplotlib; matplotlib.use("Agg")
from runner.posttrain import Trainer  # noqa: F401
print("runner.posttrain import OK")
PYEOF

log "环境完成。下一步：source ${HERE}/env.sh"
