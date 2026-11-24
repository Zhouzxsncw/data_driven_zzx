#!/usr/bin/env bash
# NVIDIA / 江算 PPU 宿主机：在已有 torch 上装 LingBot-VLA 依赖，不覆盖 torch。
# 用法（在目标机、代码目录下）：
#   bash recipes/lingbot_vla/setup_host_env.sh
# 或：
#   LINGBOT_SRC=/path/to/lingbot-vla-v2 VENV=/path/to/venv bash setup_host_env.sh
set -euo pipefail

SRC="${LINGBOT_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# 上面在 recipes/lingbot_vla 时会指到 data_Driven。允许直接传仓库根。
if [ ! -f "${SRC}/setup.py" ] && [ -f "${LINGBOT_SRC:-}/setup.py" ]; then
  SRC="${LINGBOT_SRC}"
fi
if [ ! -f "${SRC}/setup.py" ]; then
  echo "[ERR] lingbot-vla-v2 setup.py not found at SRC=${SRC}" >&2
  exit 1
fi
VENV="${VENV:-${SRC}/.venv}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 江算 PPU：沿用 openpi 的 env，拿到 libhggcrt / maca。
for f in \
  /home/ubuntu/cigai_train/openpi-jiangsuan/scripts/env_ppu.sh \
  /home/ubuntu/lerobot/CQTongBot-train-ppu/scripts/env_ppu.sh \
  "${SRC}/scripts/env_ppu.sh"; do
  if [ -f "$f" ]; then
    # shellcheck disable=SC1090
    source "$f"
    break
  fi
done

if [ ! -x "${VENV}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONNOUSERSITE=1

python -m pip install -U pip setuptools wheel -i "${PIP_INDEX}"

# 不碰已有 torch。flash-attn 有 wheel 才装。
python -m pip install \
  "transformers==4.57.3" \
  "tokenizers==0.22.2" \
  "datasets==3.6.0" \
  "huggingface-hub==0.34.3" \
  "safetensors==0.5.3" \
  "qwen-vl-utils==0.0.11" \
  "peft==0.15.2" \
  "numpy==1.26.4" \
  "accelerate==1.7.0" \
  "einops==0.8.1" \
  "omegaconf==2.3.0" \
  "opencv-python-headless==4.11.0.86" \
  "pillow==11.3.0" \
  "sentencepiece==0.2.0" \
  "protobuf==3.20.3" \
  "tqdm==4.67.1" \
  "pyyaml==6.0.2" \
  "av==15.0.0" \
  "imageio==2.34.2" \
  "h5py==3.14.0" \
  "jsonlines==4.0.0" \
  "rich==14.1.0" \
  -i "${PIP_INDEX}"

python -m pip install --no-deps -e "${SRC}" -i "${PIP_INDEX}"
DEPTH="${SRC}/lingbotvla/models/vla/vision_models"
if [ -d "${DEPTH}/lingbot-depth" ]; then
  python -m pip install --no-deps -e "${DEPTH}/lingbot-depth" -i "${PIP_INDEX}" || true
fi
if [ -d "${DEPTH}/MoGe" ]; then
  python -m pip install --no-deps -e "${DEPTH}/MoGe" -i "${PIP_INDEX}" || true
fi

if [ -n "${FLASH_ATTN_WHEEL:-}" ] && [ -f "${FLASH_ATTN_WHEEL}" ]; then
  python -m pip install --no-deps "${FLASH_ATTN_WHEEL}" || true
fi

python - << 'PY'
import torch, transformers, lingbotvla
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "ngpu", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("transformers", transformers.__version__)
print("lingbotvla", lingbotvla.__file__)
assert transformers.__version__.startswith("4.57"), transformers.__version__
print("HOST_ENV_OK")
PY
echo "[OK] host env ready: ${VENV}"
