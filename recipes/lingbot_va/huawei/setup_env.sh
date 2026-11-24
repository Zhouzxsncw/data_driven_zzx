#!/bin/bash
# 在 lingbot-va 容器内执行。保留镜像 torch_npu，不装 CUDA torch / flash-attn / triton。
set -euo pipefail
ROOT="${LINGBOT_VA_ROOT:-/home/lingbot-va}"
SRC="${LINGBOT_VA_CODE:-${ROOT}/src/lingbot-va}"
VENV="${VENV:-${ROOT}/venv}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTHONPATH="${SRC}:${ROOT}/shim:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export PYTHONNOUSERSITE=1
mkdir -p "${ROOT}/shim" "${ROOT}/hf" "${ROOT}/outputs" "${ROOT}/logs"

if [ ! -x "${VENV}/bin/python" ]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python - << 'PY'
import torch, torch_npu
print("image-torch", torch.__version__, "torch_npu", torch_npu.__version__, "npu", torch.npu.is_available())
assert torch.npu.is_available(), "NPU not visible"
PY

python -m pip install -U pip "setuptools>=71,<80" wheel -i "${PIP_INDEX}"

# 不装 torch / torchvision / torchaudio / triton / flash-attn。
# --no-deps：避免 peft/diffusers 把镜像 torch_npu 换成 CPU torch。
python -m pip install --ignore-installed --no-deps \
  "diffusers==0.36.0" \
  "transformers==4.55.2" \
  "tokenizers==0.21.4" \
  "huggingface-hub==0.34.3" \
  "safetensors==0.5.3" \
  "einops==0.8.1" \
  "easydict==1.13" \
  "ftfy==6.3.1" \
  "numpy==1.26.4" \
  "accelerate==1.7.0" \
  "sentencepiece==0.2.0" \
  "protobuf==3.20.3" \
  "regex==2024.11.6" \
  "tqdm==4.67.1" \
  "pyyaml==6.0.2" \
  "pillow==11.3.0" \
  "packaging==25.0" \
  -i "${PIP_INDEX}"

python - << PY
import torch, torch_npu, diffusers, transformers, einops, easydict
print("torch", torch.__version__, "npu", torch.npu.is_available(), "ndev", torch.npu.device_count())
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
assert torch.npu.is_available()
import torch_npu
print("torch_npu", torch_npu.__version__)
PY
echo "[OK] lingbot-va ascend env ready: ${VENV}"
