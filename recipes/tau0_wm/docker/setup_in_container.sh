#!/bin/bash
# 在 tau0wm 容器内执行。保留镜像里的 PPU / NPU / CUDA torch，不装 flash-attn，不换 torch。
# ROOT：江算/天翼云/H100 为 /workspace；华为为 /home/tau0wm。
set -euo pipefail

ROOT="${ROOT:-${TAU_ROOT:-/workspace}}"
VENV="${VENV:-${ROOT}/venv}"
REPO="${REPO:-${ROOT}/repo/tau-0-wm}"
CODE="${CODE:-${ROOT}/code}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
LOG="${ROOT}/logs/setup_in_container.log"

mkdir -p "${ROOT}/logs" "${ROOT}/hf" "${ROOT}/pretrained" "${ROOT}/outputs" "${ROOT}/shim"
: > "${LOG}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

set +u
# shellcheck disable=SC1091
# 天翼云 zz_ppucompat 必须单独生效。再 source 镜像里的 envsetup.sh 会把旧 SDK 1.6.1
# 盖回来，ppu2.1.0 的 torch 会报 ncclCommInitRankScalable。
if [ -f /etc/profile.d/zz_ppucompat.sh ]; then
  source /etc/profile.d/zz_ppucompat.sh
elif [ -f /usr/local/PPU_SDK/envsetup.sh ]; then
  source /usr/local/PPU_SDK/envsetup.sh
fi
for f in /usr/local/Ascend/ascend-toolkit/set_env.sh \
         /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh \
         /usr/local/Ascend/nnal/atb/set_env.sh; do
  [ -f "$f" ] && source "$f"
done
set -u

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-${ROOT}/hf}"
export PYTHONPATH="${REPO}:${ROOT}/shim${PYTHONPATH:+:${PYTHONPATH}}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [ ! -x "${VENV}/bin/python" ]; then
  log "创建 venv --system-site-packages ${VENV}"
  python3 -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
log "python=$(command -v python) $(python -V 2>&1)"

python - << 'PY' | tee -a "${LOG}"
import torch
print("image_torch", torch.__version__, torch.__file__)
PY

if [ -f "${ROOT}/shim/sitecustomize.py" ] || [ -f "${CODE}/huawei/sitecustomize.py" ]; then
  src="${ROOT}/shim/sitecustomize.py"
  [ -f "${src}" ] || src="${CODE}/huawei/sitecustomize.py"
  sp="$(python -c 'import site; print(site.getsitepackages()[0])')"
  cp -f "${src}" "${sp}/sitecustomize.py"
  log "sitecustomize -> ${sp}"
fi

log "pip 主依赖（不指定 torch）"
python -m pip install -U pip "setuptools<81" wheel -i "${PIP_INDEX}" >>"${LOG}" 2>&1

# 这些包会声明 torch 依赖。--no-deps 避免把镜像 torch 盖成 PyPI CUDA 轮。
python -m pip install --no-deps -i "${PIP_INDEX}" \
  "accelerate==1.12.0" \
  "diffusers==0.35.2" \
  "transformers==4.57.6" \
  "kornia==0.8.2" \
  "kornia-rs" \
  "datasets==4.1.1" \
  >>"${LOG}" 2>&1

python -m pip install -i "${PIP_INDEX}" \
  "tokenizers==0.22.2" \
  "huggingface-hub==0.35.3" \
  "safetensors==0.6.2" \
  "pyarrow==23.0.1" \
  "pandas==2.3.1" \
  "einops==0.8.1" \
  "omegaconf==2.3.0" \
  "PyYAML==6.0.2" \
  "pyyaml-include==1.4.1" \
  "av==15.1.0" \
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
  packaging termcolor jsonlines deepdiff \
  "draccus==0.10.0" dm-tree gymnasium \
  regex fsspec aiohttp dill multiprocess xxhash \
  python-dateutil pytz wcwidth decorator proglog \
  >>"${LOG}" 2>&1 || log "WARN 主依赖有冲突，见日志"

python -m pip install --no-deps -i "${PIP_INDEX}" "lerobot==0.4.4" >>"${LOG}" 2>&1 \
  || log "WARN lerobot 安装异常"

# venv 里若被塞进第二份 torch / nvidia，删掉，回到镜像那份。
sp="$(python - << 'PY'
import site
hits = [p for p in site.getsitepackages() if "/venv/" in p or p.rstrip("/").endswith("/venv")]
print(hits[0] if hits else "")
PY
)"
if [ -n "${sp}" ] && [ -d "${sp}" ]; then
  rm -rf "${sp}/torch" "${sp}/torchgen" "${sp}/functorch" "${sp}/nvidia" \
    "${sp}"/torch-*.dist-info "${sp}"/nvidia_*.dist-info || true
  log "已清掉 venv 内影子 torch: ${sp}"
fi

python - << 'PY' | tee -a "${LOG}"
import torch
print("torch_after", torch.__version__, torch.__file__)
assert "site-packages/torch" in torch.__file__
PY

if [ -f "${CODE}/patch_tau_repo.py" ] && [ -d "${REPO}/runner" ]; then
  log "打 fork 补丁 ${REPO}"
  PYTHONPATH="${REPO}" python "${CODE}/patch_tau_repo.py" --repo "${REPO}" >>"${LOG}" 2>&1 \
    || log "WARN patch 未完全通过，见日志"
fi

# Wan2.2 三件：机器上已有就软链，不重复下 32G。
PRE="${ROOT}/pretrained"
WAN_DST="${PRE}/Wan2.2-TI2V-5B"
if [ ! -f "${WAN_DST}/Wan2.2_VAE.pth" ]; then
  found=""
  for cand in \
    /data1/dreamzero-pretrained/Wan2.2-TI2V-5B \
    /nvme1n1/dreamzero-pretrained/Wan2.2-TI2V-5B \
    /nvme2n1/dreamzero-pretrained/Wan2.2-TI2V-5B \
    /data/extra/dreamzero-pretrained/Wan2.2-TI2V-5B \
    /data/dreamzero-pretrained/Wan2.2-TI2V-5B \
    /home/dreamzero-pretrained/Wan2.2-TI2V-5B
  do
    if [ -f "${cand}/Wan2.2_VAE.pth" ] && [ -f "${cand}/models_t5_umt5-xxl-enc-bf16.pth" ]; then
      found="${cand}"
      break
    fi
  done
  if [ -n "${found}" ]; then
    mkdir -p "${WAN_DST}/google"
    ln -sfn "${found}/models_t5_umt5-xxl-enc-bf16.pth" "${WAN_DST}/models_t5_umt5-xxl-enc-bf16.pth"
    ln -sfn "${found}/Wan2.2_VAE.pth" "${WAN_DST}/Wan2.2_VAE.pth"
    ln -sfn "${found}/google/umt5-xxl" "${WAN_DST}/google/umt5-xxl"
    log "Wan2.2 软链 ${found}"
  else
    log "WARN 没找到 Wan2.2 T5/VAE，训练前要另补"
  fi
fi

export TAU_REPO="${REPO}"
export SMOKE_DEV="${SMOKE_DEV:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
log "smoke SMOKE_DEV=${SMOKE_DEV}"
python "${CODE}/docker/smoke_test.py" | tee -a "${LOG}"
touch "${ROOT}/SETUP_DONE"
log "SETUP_DONE ${ROOT}"
