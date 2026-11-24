#!/bin/bash
# 在 wall-oss 容器内执行。用镜像自带 torch_npu，不覆盖 torch，跳过 CUDA ext。
set -euo pipefail
ROOT="${WALLOSS_ROOT:-/home/wall-oss}"
VENV="${VENV:-${ROOT}/venv}"
WALLX="${WALLX:-${ROOT}/src/wall-x}"
LEROBOT="${LEROBOT:-${ROOT}/src/lerobot}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export WALLX_SKIP_CUDA_EXT=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT}/hf}"
mkdir -p "${ROOT}/shim" "${HF_HOME}" "${ROOT}/outputs" "${ROOT}/logs"

if [ ! -x "${VENV}/bin/python" ]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${WALLX}:${LEROBOT}:${ROOT}/shim:${PYTHONPATH:-}"
python "${ROOT}/code/huawei/patch_skip_flash.py" "${WALLX}" || true

if python - << 'PY'
import transformers
import torch_npu
import torch
import wall_x
from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
assert transformers.__version__.startswith("5.2")
assert torch.npu.is_available()
print("skip-pip transformers", transformers.__version__, "torch_npu", torch_npu.__version__, "ndev", torch.npu.device_count())
print("wall_x", wall_x.__file__)
PY
then
  echo "[OK] env already usable, skip pip"
  exit 0
fi

python -m pip install -U pip "setuptools>=71,<80" wheel -i "${PIP_INDEX}"

# 不装 torch / torchvision / torchaudio / flash-attn。
python -m pip install \
  "transformers==5.2.0" \
  "accelerate==1.13.0" \
  "peft==0.18.1" \
  "numpy>=1.26,<2" \
  "safetensors>=0.8,<0.9" \
  "einops" \
  "pillow>=10,<12" \
  "pyyaml>=6.0" \
  "scipy==1.15.3" \
  "tqdm>=4.67,<5" \
  "qwen-vl-utils==0.0.11" \
  "torchdiffeq==0.2.5" \
  "websockets>=12.0,<16" \
  "msgpack-numpy>=0.4.8,<0.5" \
  "opencv-python-headless>=4.11,<4.12" \
  -i "${PIP_INDEX}"

if [ -f "${LEROBOT}/pyproject.toml" ] || [ -f "${LEROBOT}/setup.py" ]; then
  python -m pip install --no-deps -e "${LEROBOT}" -i "${PIP_INDEX}" || true
fi
if [ -f "${WALLX}/setup.py" ]; then
  WALLX_SKIP_CUDA_EXT=1 python -m pip install --no-build-isolation --no-deps -e "${WALLX}" \
    -i "${PIP_INDEX}"
fi

python - << 'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
assert transformers.__version__.startswith("5.2"), transformers.__version__
import torch_npu
print("torch_npu", torch_npu.__version__, "npu", torch.npu.is_available(), "ndev", torch.npu.device_count())
assert torch.npu.is_available(), "NPU not visible"
import wall_x
print("wall_x", getattr(wall_x, "__file__", "?"))
from wall_x.model.core.ops import permute, rmsnorm
print("ops fallback OK", type(permute).__name__, type(rmsnorm).__name__)
from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
print("Qwen2_5_VLMoEForAction OK", Qwen2_5_VLMoEForAction)
PY
echo "[OK] wall-oss ascend env ready: ${VENV}"
