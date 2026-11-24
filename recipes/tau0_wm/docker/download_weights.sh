#!/bin/bash
# 容器内或宿主机：hf-mirror 续传 sii-research/tau-0-wm（约 23G）。已齐则退出。
set -euo pipefail
ROOT="${ROOT:-${TAU_ROOT:-/workspace}}"
PRE="${ROOT}/pretrained"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
mkdir -p "${PRE}/tau-0-wm/simulator" "${ROOT}/logs"
LOG="${ROOT}/logs/hf_download.log"
if [ -f "${PRE}/tau-0-wm/simulator/diffusion_pytorch_model.bin" ] \
   && [ -f "${PRE}/tau-0-wm/diffusion_pytorch_model-00001-of-00002.bin" ]; then
  echo "WEIGHTS_PRESENT"
  exit 0
fi
HF_BASE="${HF_ENDPOINT}/sii-research/tau-0-wm/resolve/main"
pull() {
  local rel="$1" dest="$2" min_bytes="$3"
  if [ -f "${dest}" ] && [ "$(stat -c%s "${dest}")" -ge "${min_bytes}" ]; then
    echo "[skip] ${rel}"
    return 0
  fi
  echo "[get] ${rel}"
  curl -fL --retry 8 --retry-delay 3 \
    -C - --connect-timeout 20 \
    -o "${dest}" "${HF_BASE}/${rel}"
}
{
  pull README.md "${PRE}/tau-0-wm/README.md" 100
  pull config.json "${PRE}/tau-0-wm/config.json" 100
  pull diffusion_pytorch_model.bin.index.json \
    "${PRE}/tau-0-wm/diffusion_pytorch_model.bin.index.json" 1000
  pull simulator/config.json "${PRE}/tau-0-wm/simulator/config.json" 100
  pull diffusion_pytorch_model-00002-of-00002.bin \
    "${PRE}/tau-0-wm/diffusion_pytorch_model-00002-of-00002.bin" 1000000000
  pull diffusion_pytorch_model-00001-of-00002.bin \
    "${PRE}/tau-0-wm/diffusion_pytorch_model-00001-of-00002.bin" 9000000000
  pull simulator/diffusion_pytorch_model.bin \
    "${PRE}/tau-0-wm/simulator/diffusion_pytorch_model.bin" 11000000000
  echo WEIGHTS_DONE
} > "${LOG}" 2>&1
echo "WEIGHTS_DONE"
