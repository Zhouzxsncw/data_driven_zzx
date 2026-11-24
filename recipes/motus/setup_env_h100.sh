#!/usr/bin/env bash
# h100-103: Motus 依赖安装（venv + torch cu128 + lerobot 0.3.2 --no-deps）
# H100 = Hopper sm_90（不是 Blackwell sm_120）
# 用法: bash /data1/motus/recipes/setup_env_h100.sh
set -euo pipefail

VENV="${VENV:-/data1/motus-env}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-${HERE}/setup_env_h100.log}"

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TORCH_CUDA_ARCH_LIST="9.0"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

log "========== Motus H100 setup =========="
log "VENV=${VENV}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  log "creating venv (python3.10)"
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install -U pip setuptools wheel packaging ninja

if python -c "import torch" 2>/dev/null; then
  log "torch already present: $(python -c 'import torch; print(torch.__version__)')"
else
  # 本机 pip 下载大文件仅 ~0.9MB/s（Python HTTP 栈被限速），但 curl 直链 ~12MB/s；
  # 故 torch/torchvision + 全部 nvidia 依赖（cudnn 571M/cublas 等共 2.7G）预先 curl 下到
  # /data1/motus/wheels/，这里目录内全部 wheel 一次安装；仅 sympy/networkx/jinja2 等
  # 小依赖（几 MB）走 aliyun index
  WHEELS_DIR="${WHEELS_DIR:-/data1/motus/wheels}"
  if ls "${WHEELS_DIR}"/torch-2.7.1-*.whl >/dev/null 2>&1; then
    log "install torch 2.7.1 + nvidia deps from local wheels (${WHEELS_DIR})"
    # 一次传入目录内全部 wheel 文件路径（--find-links 会被 index 下载抢先）
    python -m pip install "${WHEELS_DIR}"/*.whl
  else
    log "install torch 2.7.1 from aliyun pypi (fallback)"
    python -m pip install torch==2.7.1 torchvision==0.22.1
  fi
fi

python - <<'PY' | tee -a "$LOG_FILE"
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"gpu0={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
PY

log "install Motus python deps"
python -m pip install \
  "huggingface-hub>=0.34.0" \
  "hf_transfer" \
  "modelscope>=1.20.0" \
  "qwen-vl-utils>=0.0.11" \
  "tokenizers>=0.15.0" \
  "accelerate>=0.25.0" \
  "diffusers>=0.25.0" \
  "opencv-python>=4.8.0" \
  "Pillow>=10.0.0" \
  "imageio>=2.31.0" \
  "imageio-ffmpeg>=0.4.8" \
  "numpy>=1.24.0,<2.3" \
  "h5py>=3.9.0" \
  "scipy>=1.11.0" \
  "pyyaml>=6.0" \
  "wandb>=0.16.0" \
  "tensorboard>=2.15.0" \
  "tqdm>=4.66.0" \
  "matplotlib>=3.7.0" \
  "seaborn>=0.12.0" \
  "omegaconf>=2.3.0" \
  "easydict>=1.13" \
  "ftfy>=6.3.1" \
  "safetensors" \
  "einops" \
  "sentencepiece" \
  "protobuf" \
  "av" \
  "jsonlines" \
  "pyarrow"

# Qwen3-VL 需要较新 transformers；官方 pin 5.0.0rc0，国内源没有则回退 4.57+
if ! python -c "import transformers; from transformers import Qwen3VLForConditionalGeneration" 2>/dev/null; then
  log "install transformers (Qwen3-VL)"
  python -m pip install "transformers==5.0.0rc0" \
    || python -m pip install "transformers>=4.57.0"
fi

python -m pip install "deepspeed>=0.18.3" || log "WARN deepspeed pip failed"
python -m pip install "decord>=0.6.0" || log "WARN decord skipped"

# lerobot 0.3.2 必须严格 pin + --no-deps（0.5+ 弃用旧 import 路径）
if ! python -c "import lerobot" 2>/dev/null; then
  log "install lerobot==0.3.2 --no-deps + 子依赖"
  python -m pip install --no-deps "lerobot==0.3.2"
  python -m pip install "datasets==2.19.0" "draccus==0.10.0" "deepdiff>=7.0.1,<9.0.0" "rerun-sdk>=0.21.0,<0.23.0" || true
fi

log "========== core env ready（flash-attn 可后台补装） =========="

# flash-attn：先试 pip cache 命中（cosmos-env 装过 2.8.3.post1 cp310），否则源码编译；
# 192 核机器 MAX_JOBS 提速；失败不阻断（PyTorch SDPA 在 H100 上已是 flash 内核）
if ! python -c "import flash_attn" 2>/dev/null; then
  log "install flash-attn (pip cache or source build sm_90)"
  export CUDA_HOME=/usr/local/cuda-12.4
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export MAX_JOBS=32
  python -m pip install "flash-attn==2.8.3.post1" --no-build-isolation \
    || python -m pip install flash-attn --no-build-isolation \
    || log "WARN flash-attn skipped; H100 SDPA 内核可用，速度影响小"
fi

log "========== import check =========="
python - <<'PY' | tee -a "$LOG_FILE"
import importlib
ok = True
for m in ["torch", "torchvision", "transformers", "accelerate", "diffusers", "deepspeed", "omegaconf", "lerobot", "datasets"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"FAIL {m}: {e}")
        ok = False
try:
    from transformers import Qwen3VLForConditionalGeneration
    print("OK Qwen3VLForConditionalGeneration")
except Exception as e:
    print(f"FAIL Qwen3VL: {e}")
    ok = False
try:
    # 训练代码实际需要的 import 面（CODEBASE_VERSION 在 0.3.2 布局中不存在，别 check 它）
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset
    from lerobot.datasets.video_utils import decode_video_frames, VideoFrame
    print("OK lerobot train imports (dataset trio + video_utils)")
except Exception as e:
    print(f"FAIL lerobot train imports: {e}")
    ok = False
try:
    import flash_attn
    print(f"OK flash_attn {getattr(flash_attn, '__version__', '')}")
except Exception as e:
    print(f"WARN flash_attn: {e}")
if not ok:
    raise SystemExit(1)
PY

log "========== setup done =========="
