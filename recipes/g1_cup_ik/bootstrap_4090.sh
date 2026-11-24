#!/usr/bin/env bash
# 在松龄 4090 上装 Python 3.12 + torch cu12，给 GR00T N1.7 微调用。
# 不要从 gpu-master 拷 .conda / .venv（那是 Blackwell sm_120）。
# Isaac Sim / Isaac Lab conda 采集环境另装，本脚本不碰。
set -euo pipefail
ROOT="${UNITREE_ROOT:-${HOME}/unitree_g1}"
GR00T_DIR="${GR00T_DIR:-${HOME}/VLA/Isaac-GR00T}"
if [ ! -d "${GR00T_DIR}/gr00t" ]; then
    GR00T_DIR="${ROOT}/VLA/Isaac-GR00T"
fi
MC="${HOME}/miniconda3"
ENV_PREFIX="${GR00T_DIR}/.venv"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
echo "[bootstrap] gpu=${gpu_name} gr00t=${GR00T_DIR}"
if ! printf '%s' "${gpu_name}" | grep -qi '4090'; then
    echo "[bootstrap] 这台不是 4090，停。不要在本机 96GB 卡上跑这个脚本。" >&2
    exit 1
fi
if [ ! -d "${GR00T_DIR}/gr00t" ]; then
    echo "[bootstrap] 找不到 Isaac-GR00T：${GR00T_DIR}" >&2
    exit 1
fi

if [ ! -x "${MC}/bin/conda" ]; then
    echo "[bootstrap] 装 Miniconda -> ${MC}"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${MC}"
fi
# shellcheck disable=SC1091
source "${MC}/etc/profile.d/conda.sh"

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    echo "[bootstrap] conda env python 3.12 -> ${ENV_PREFIX}"
    conda create -y -p "${ENV_PREFIX}" python=3.12 pip
fi
conda activate "${ENV_PREFIX}"
pip install --upgrade pip

if ! python -c "import torch" 2>/dev/null; then
    echo "[bootstrap] pip torch 2.9 cu126（4090 sm_89）"
    pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
fi

cd "${GR00T_DIR}"
export MAX_JOBS="${MAX_JOBS:-2}"
echo "[bootstrap] 最小依赖（跳过 tensorrt；flash-attn 能装再装）"
pip install huggingface-hub 'opencv-python-headless>=4.5,<4.13' 'diffusers==0.35.1' dm-tree \
    'lmdb==1.7.5' 'msgpack==1.1.0' 'msgpack-numpy==0.4.8' 'pandas==2.2.3' \
    'peft==0.17.1' 'transformers==4.57.3' 'tyro==0.9.17' 'datasets==3.6.0' \
    'einops==0.8.1' 'numpy==1.26.4' 'omegaconf==2.3.0' 'scipy==1.15.3' \
    'pyzmq==27.0.1' 'wandb==0.23.0' 'albumentations==1.4.18' 'gymnasium==1.2.2' \
    'matplotlib==3.10.1' 'jsonlines==4.0.0' 'gitpython==3.1.50' 'click==8.1.8' \
    'termcolor==3.2.0' 'cryptography>=46.0.7'
pip install -e . --no-deps
pip install 'flash-attn==2.8.3' --no-build-isolation || echo "[bootstrap] flash-attn 跳过（训练还能跑，会慢）"

python - <<'PY'
import torch
print("[bootstrap] torch", torch.__version__, "cuda", torch.version.cuda,
      "cap", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "CUDA 不可用"
maj, minor = torch.cuda.get_device_capability(0)
assert (maj, minor) == (8, 9), f"期望 sm_89，实际 {maj}.{minor}"
import gr00t
print("[bootstrap] GR00T 训练环境就绪。采集/评测还要另装 Isaac Sim（不要拷本机 env_isaaclab）。")
PY
