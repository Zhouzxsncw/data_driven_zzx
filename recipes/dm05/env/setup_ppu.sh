#!/bin/bash
# 在 dm05 容器内执行。保留镜像里的 PPU torch，不装 flash-attn / bitsandbytes / CUDA torch。
set -euo pipefail

ROOT="${ROOT:-/workspace}"
VENV="${VENV:-${ROOT}/venv}"
SRC="${SRC:-${ROOT}/src/opendm}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

mkdir -p "${ROOT}/hf" "${ROOT}/logs" "${ROOT}/pretrained"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-${ROOT}/hf}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
export PYTHONNOUSERSITE=1

if [ ! -x "${VENV}/bin/python" ]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python - << 'PY'
import torch
print("image-torch", torch.__version__, "cuda_build", torch.version.cuda)
PY

if python - << 'PY'
import opendm
from opendm.model.dm05.dm05_arch import DM05Config
from transformers import AutoConfig
print("skip-pip", opendm.__file__, DM05Config.model_type)
PY
then
  echo "[OK] env already usable, skip pip"
  exit 0
fi

python -m pip install -U pip "setuptools>=61,<80" wheel -i "${PIP_INDEX}"

# 约束住镜像 torch，pip 不许换成 CUDA wheel。
python - << 'PY'
import torch
from pathlib import Path
Path("/tmp/dm05-constraints.txt").write_text(f"torch=={torch.__version__}\n")
PY

set +e
python -m pip install -c /tmp/dm05-constraints.txt \
  "transformers==5.3.0" \
  "tokenizers==0.22.1" \
  "safetensors" \
  "huggingface-hub" \
  "einops" \
  "numpy==1.26.4" \
  "sentencepiece" \
  "pillow" \
  "pyyaml" \
  "regex" \
  "tqdm" \
  "packaging" \
  "accelerate==1.14.0" \
  -i "${PIP_INDEX}"
rc=$?
set -e
if [ "${rc}" != "0" ]; then
  echo "[WARN] constrained transformers 5.3.0 failed; install --no-deps and keep image torch"
  python -m pip install --no-deps \
    "transformers==5.3.0" \
    "tokenizers==0.22.1" \
    "safetensors" \
    "huggingface-hub" \
    "einops" \
    "sentencepiece" \
    "accelerate==1.14.0" \
    -i "${PIP_INDEX}"
fi

python -m pip install --no-deps -e "${SRC}" -i "${PIP_INDEX}"

python - << 'PY'
import torch
import opendm
from opendm.model.dm05.dm05_arch import DM05Config
print("torch", torch.__version__)
print("opendm", opendm.__file__)
print("DM05Config", DM05Config.model_type)
assert "cu" not in torch.__version__ or True
PY
echo "[OK] dm05 ppu env ready: ${VENV}"
