#!/bin/bash
# 在 lingbot-vla 容器内执行。保留镜像 torch_npu，不装 CUDA torch / flash-attn / triton。
set -euo pipefail
ROOT="${LINGBOT_ROOT:-/home/lingbot-vla}"
SRC="${LINGBOT_SRC:-${ROOT}/src/lingbot-vla-v2}"
VENV="${VENV:-${ROOT}/venv}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT}/hf}"
export PYTHONPATH="${SRC}:${ROOT}/shim:${PYTHONPATH:-}"
mkdir -p "${ROOT}/shim" "${HF_HOME}" "${ROOT}/outputs" "${ROOT}/logs"

if [ ! -x "${VENV}/bin/python" ]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if python - << 'PY'
import transformers
ok = transformers.__version__.startswith("4.57")
print("transformers", transformers.__version__, "ok" if ok else "need-pin")
raise SystemExit(0 if ok else 1)
PY
then
  if python - << 'PY'
import torch, torch_npu, lingbotvla
assert torch.npu.is_available()
print("skip-pip torch", torch.__version__, "torch_npu", torch_npu.__version__,
      "ndev", torch.npu.device_count(), "lingbotvla", lingbotvla.__file__)
PY
  then
    echo "[OK] env already usable, skip pip"
    exit 0
  fi
fi

python -m pip install -U pip "setuptools>=71,<80" wheel -i "${PIP_INDEX}"

# 不装 torch / torchvision / torchaudio / torchdata / torchcodec / triton / flash-attn。
# --no-deps：peft/accelerate 会把镜像 torch_npu 换成 CPU torch。
python -m pip install --ignore-installed --no-deps \
  "transformers==4.57.3" \
  "tokenizers==0.22.2" \
  "datasets==3.6.0" \
  "huggingface-hub==0.34.3" \
  "safetensors==0.5.3" \
  "qwen-vl-utils==0.0.11" \
  "peft==0.15.2" \
  "numpy==1.26.4" \
  "scipy==1.13.1" \
  "pandas==2.3.1" \
  "pyarrow==21.0.0" \
  "pillow==11.3.0" \
  "opencv-python-headless==4.11.0.86" \
  "einops==0.8.1" \
  "tqdm==4.67.1" \
  "pyyaml==6.0.2" \
  "accelerate==1.7.0" \
  "sentencepiece==0.2.0" \
  "protobuf==3.20.3" \
  "omegaconf==2.3.0" \
  "msgpack==1.1.1" \
  "websockets==15.0.1" \
  "av==15.0.0" \
  "imageio==2.34.2" \
  "h5py==3.14.0" \
  "jsonlines==4.0.0" \
  "rich==14.1.0" \
  "packaging==25.0" \
  "psutil==7.0.0" \
  -i "${PIP_INDEX}"

python -m pip install --no-deps -e "${SRC}" -i "${PIP_INDEX}"

DEPTH="${SRC}/lingbotvla/models/vla/vision_models"
if [ -d "${DEPTH}/lingbot-depth" ]; then
  python -m pip install --no-deps -e "${DEPTH}/lingbot-depth" -i "${PIP_INDEX}" || true
fi
if [ -d "${DEPTH}/MoGe" ]; then
  python -m pip install --no-deps -e "${DEPTH}/MoGe" -i "${PIP_INDEX}" || true
fi

python - << 'PY'
import torch, transformers, lingbotvla
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
assert transformers.__version__.startswith("4.57"), transformers.__version__
import torch_npu
print("torch_npu", torch_npu.__version__, "npu", torch.npu.is_available(), "ndev", torch.npu.device_count())
assert torch.npu.is_available(), "NPU not visible"
print("lingbotvla", lingbotvla.__file__)
PY
echo "[OK] lingbot-vla ascend env ready: ${VENV}"
