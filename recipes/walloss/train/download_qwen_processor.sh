#!/bin/bash
# 下载 Qwen2.5-VL-3B-Instruct 的 processor/tokenizer（跳过权重）。
# 容器内: bash /workspace/code/train/download_qwen_processor.sh
set -euo pipefail
ROOT="${ROOT:-/workspace}"
DST="${DST:-${ROOT}/pretrained/Qwen2.5-VL-3B-Instruct}"
REPO="${REPO:-Qwen/Qwen2.5-VL-3B-Instruct}"
VENV="${VENV:-${ROOT}/venv}"

set +u
source /usr/local/PPU_SDK/envsetup.sh
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
set -u

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT}/hf}"
mkdir -p "${DST}" "${HF_HOME}"

if [ -f "${DST}/preprocessor_config.json" ] && [ -f "${DST}/tokenizer.json" ]; then
  echo "[OK] processor already present: ${DST}"
  exit 0
fi

python - << PY
import os
from huggingface_hub import snapshot_download

dst = os.environ.get("DST", "${DST}")
repo = "${REPO}"
patterns = [
    "*.json",
    "*.txt",
    "*.jinja",
    "*.model",
    "tokenizer*",
    "vocab*",
    "merges*",
    "preprocessor*",
    "chat_template*",
    "special_tokens*",
    "added_tokens*",
    "generation_config*",
    "video_preprocessor*",
    "README*",
    ".gitattributes",
]
print("[INFO] downloading processor files", repo, "->", dst)
try:
    snapshot_download(
        repo,
        local_dir=dst,
        allow_patterns=patterns,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.npz"],
    )
except Exception as e:
    print("hf failed:", e)
    from modelscope import snapshot_download as ms_dl
    ms_dl("Qwen/Qwen2.5-VL-3B-Instruct", local_dir=dst)
print("download done")
PY

test -f "${DST}/preprocessor_config.json"
test -f "${DST}/tokenizer.json" || test -f "${DST}/tokenizer_config.json"
echo "[OK] ${DST}"
ls -lh "${DST}" | head -40
