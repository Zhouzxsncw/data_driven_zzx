#!/bin/bash
# 在 cosmos3 容器内执行。保留镜像 torch/PPU，不装 flash-attn / transformer-engine / CUDA torch。
set -euo pipefail

ROOT="${ROOT:-/workspace}"
VENV="${VENV:-${ROOT}/venv}"
CODE="${CODE:-${ROOT}/code}"
SRC="${SRC:-${ROOT}/src/cosmos-framework}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

set +u
# PPU 镜像才有；华为容器没有这文件，忽略。
# shellcheck disable=SC1091
source /usr/local/PPU_SDK/envsetup.sh 2>/dev/null || true
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT}/hf}"
export I4_ATTN_BACKENDS="${I4_ATTN_BACKENDS:-sdpa}"
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${SRC}:${ROOT}/shim:${PYTHONPATH:-}"
mkdir -p "${ROOT}" "${HF_HOME}" "${ROOT}/outputs" "${ROOT}/src" "${ROOT}/logs" "${ROOT}/shim"

if [ ! -x "${VENV}/bin/python" ]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python "${CODE}/env/patch_sdpa_backend.py" "${SRC}"

if python - << 'PY'
import cosmos_framework
from cosmos_framework.model.attention.frontend import BACKEND_MAP
assert "sdpa" in BACKEND_MAP, sorted(BACKEND_MAP)
print("skip-pip cosmos_framework", cosmos_framework.__file__, "backends", sorted(BACKEND_MAP))
PY
then
  echo "[OK] env already usable, skip pip"
  exit 0
fi

python -m pip install -U pip "setuptools>=71,<80" wheel -i "${PIP_INDEX}"

# 不装 torch / flash-attn / transformer-engine / nvidia-cudnn-frontend。
python -m pip install --ignore-installed --no-deps \
  "transformers==4.57.3" \
  "tokenizers==0.22.2" \
  "accelerate==1.7.0" \
  "huggingface-hub==0.34.3" \
  "safetensors==0.5.3" \
  "einops==0.8.1" \
  "omegaconf==2.3.0" \
  "hydra-core==1.3.2" \
  "pydantic>=2.0" \
  "loguru" \
  "tyro" \
  "av" \
  "imageio" \
  "imageio-ffmpeg" \
  "pillow" \
  "scipy" \
  "msgpack" \
  "websockets" \
  "cattrs" \
  "tqdm" \
  "pyyaml" \
  "numpy==1.26.4" \
  -i "${PIP_INDEX}"

python -m pip install --no-deps -e "${SRC}" -i "${PIP_INDEX}"

python - << 'PY'
import torch
import cosmos_framework
from cosmos_framework.model.attention.frontend import BACKEND_MAP
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("cosmos_framework", cosmos_framework.__file__)
print("backends", sorted(BACKEND_MAP))
assert "sdpa" in BACKEND_MAP
PY
echo "[OK] cosmos3 ppu/npu env ready: ${VENV}"
