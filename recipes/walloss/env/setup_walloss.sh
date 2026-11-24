#!/bin/bash
# 在 wall_oss 容器内执行。装 wall-x + LeRobot，不覆盖镜像 torch / flash-attn。
# CUDA 自定义算子跳过，走官方 PyTorch fallback。
set -euo pipefail

ROOT="${ROOT:-/workspace}"
VENV="${VENV:-${ROOT}/venv}"
CODE="${CODE:-${ROOT}/code}"
WALLX="${WALLX:-${ROOT}/src/wall-x}"
LEROBOT="${LEROBOT:-${ROOT}/src/lerobot}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
CONSTRAINTS="${CONSTRAINTS:-${CODE}/env/constraints_walloss.txt}"
INSTALL_DMUON="${INSTALL_DMUON:-0}"

set +u
source /usr/local/PPU_SDK/envsetup.sh
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT}/hf}"
export WALLX_SKIP_CUDA_EXT=1
mkdir -p "${ROOT}" "${HF_HOME}" "${ROOT}/outputs" "${ROOT}/src" "${ROOT}/logs"

if [ ! -x "${VENV}/bin/python" ]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install -U pip "setuptools>=71,<80" wheel -i "${PIP_INDEX}"

clone_github() {
  local dest="$1"
  local org_repo="$2"
  local branch="${3:-main}"
  if [ -d "${dest}/.git" ]; then
    echo "[INFO] pull ${dest}"
    git -C "${dest}" pull --ff-only || true
    return 0
  fi
  if [ -f "${dest}/setup.py" ] || [ -f "${dest}/pyproject.toml" ]; then
    echo "[INFO] already present ${dest}"
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  local urls=(
    "https://github.com/${org_repo}.git"
    "https://ghfast.top/https://github.com/${org_repo}.git"
    "https://ghproxy.net/https://github.com/${org_repo}.git"
    "https://gitclone.com/github.com/${org_repo}.git"
  )
  local url
  for url in "${urls[@]}"; do
    echo "[INFO] clone -b ${branch} ${url} -> ${dest}"
    if git clone --depth 1 -b "${branch}" "${url}" "${dest}"; then
      return 0
    fi
    rm -rf "${dest}"
  done
  echo "[INFO] tarball fallback for ${org_repo}@${branch}"
  mkdir -p "${dest}"
  local tgz
  tgz="$(mktemp /tmp/wallx-XXXXXX.tgz)"
  if curl -fkL "https://ghfast.top/https://github.com/${org_repo}/archive/refs/heads/${branch}.tar.gz" -o "${tgz}" \
     || curl -fkL "https://ghproxy.net/https://github.com/${org_repo}/archive/refs/heads/${branch}.tar.gz" -o "${tgz}"; then
    tar -xzf "${tgz}" -C "${dest}" --strip-components=1
    rm -f "${tgz}"
    return 0
  fi
  rm -f "${tgz}"
  echo "[ERR] cannot fetch ${org_repo}@${branch}; rsync src from a machine that can reach GitHub"
  return 1
}

clone_github "${WALLX}" "qiaow371/wall-x" "tongbot" \
  || clone_github "${WALLX}" "X-Square-Robot/wall-x" "main"
clone_github "${LEROBOT}" "huggingface/lerobot" "main"

python "${CODE}/env/patch_setup_skip_cuda.py" "${WALLX}/setup.py"

# 官方 requirements 里的 torch==2.10 / flash-attn==2.8.3 一律不装。
python -m pip install \
  "transformers==5.2.0" \
  "accelerate==1.13.0" \
  "peft==0.18.1" \
  "numpy>=1.26,<2" \
  "datasets>=2.20,<5" \
  "huggingface-hub[cli]>=0.34.2" \
  "safetensors>=0.8,<0.9" \
  "einops" \
  "pillow>=10,<12" \
  "pyyaml>=6.0" \
  "scipy==1.15.3" \
  "tqdm>=4.67,<5" \
  "tyro>=0.9.0,<1.0" \
  "wandb>=0.23,<0.25" \
  "opencv-python-headless>=4.11,<4.12" \
  "qwen-vl-utils==0.0.11" \
  "torchdiffeq==0.2.5" \
  "websockets>=12.0,<16" \
  "msgpack-numpy>=0.4.8,<0.5" \
  "numpydantic>=1.8,<2" \
  "matplotlib>=3.10,<4" \
  "ninja>=1.11,<2" \
  "packaging>=24,<26" \
  "pyarrow>=14,<23" \
  -c "${CONSTRAINTS}" \
  -i "${PIP_INDEX}"

if [ "${INSTALL_DMUON}" = "1" ]; then
  python -m pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git" \
    -c "${CONSTRAINTS}" -i "${PIP_INDEX}" \
    && echo "[OK] dmuon installed" \
    || echo "[WARN] dmuon skip; use optimizer_type: adamw"
else
  echo "[INFO] INSTALL_DMUON=0, skip dmuon (SFT yaml uses adamw)"
fi

python -m pip install --no-deps -e "${LEROBOT}" -i "${PIP_INDEX}"
WALLX_SKIP_CUDA_EXT=1 python -m pip install --no-build-isolation --no-deps -e "${WALLX}" \
  -i "${PIP_INDEX}"

python - << 'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
print("transformers", transformers.__version__)
assert torch.cuda.is_available(), "PPU/CUDA not visible"
assert transformers.__version__.startswith("5.2"), transformers.__version__
import wall_x
print("wall_x", getattr(wall_x, "__file__", "?"))
from wall_x.model.core.ops import permute, rmsnorm
print("ops fallback import OK", type(permute).__name__, type(rmsnorm).__name__)
from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
print("Qwen2_5_VLMoEForAction OK", Qwen2_5_VLMoEForAction)
PY

echo "[OK] wall-oss env ready: ${VENV}"
echo "activate: source /usr/local/PPU_SDK/envsetup.sh && source ${VENV}/bin/activate"
