#!/bin/bash
# 拉 WALL-OSS-0.5（含 tokenizer / processor / normalizer，约 8.3GB）。
# 容器内: bash /workspace/code/download/download_walloss.sh
# 宿主机: bash /nvme1n1/wall-oss/code/download/download_walloss.sh
set -euo pipefail

NAME="${NAME:-wall_oss}"
DST_HOST="${DST_HOST:-/nvme1n1/wall-oss/pretrained/wall-oss-0.5}"
DST_CTR="${DST_CTR:-/workspace/pretrained/wall-oss-0.5}"
REPO="${REPO:-x-square-robot/wall-oss-0.5}"
VENV="${VENV:-/workspace/venv}"

in_container() {
  [ -f /.dockerenv ] || grep -qa docker /proc/1/cgroup 2>/dev/null
}

if in_container; then
  DST="${DST_CTR}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HOME="${HF_HOME:-/workspace/hf}"
  mkdir -p "${DST}" "${HF_HOME}"
  python -m pip install -q "huggingface-hub[cli]>=0.34.2" modelscope \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
else
  DST="${DST_HOST}"
  mkdir -p "${DST}" /nvme1n1/wall-oss/hf/modelscope
  if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "[ERR] container ${NAME} not running"
    exit 1
  fi
  docker exec "${NAME}" python3 -m pip install -q modelscope huggingface_hub \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

if [ -f "${DST}/config.json" ] && ls "${DST}"/*.safetensors >/dev/null 2>&1; then
  echo "[OK] already present: ${DST}"
else
  echo "[INFO] downloading ${REPO} -> ${DST}"
  if in_container; then
    python - << PY
import os
from huggingface_hub import snapshot_download
dst = "${DST_CTR}"
repo = "${REPO}"
try:
    snapshot_download(repo, local_dir=dst)
except Exception as e:
    print("hf failed:", e)
    from modelscope import snapshot_download as ms_dl
    ms_dl(repo, local_dir=dst)
print("download done")
PY
  else
    docker exec -i \
      -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
      -e MODELSCOPE_CACHE=/workspace/hf/modelscope \
      "${NAME}" python3 - << PY
from huggingface_hub import snapshot_download
dst = "${DST_CTR}"
repo = "${REPO}"
try:
    snapshot_download(repo, local_dir=dst)
except Exception as e:
    print("hf failed:", e)
    from modelscope import snapshot_download as ms_dl
    ms_dl(repo, local_dir=dst)
print("download done")
PY
  fi
fi

python3 - << PY || true
from pathlib import Path
p = Path("${DST}")
need = ["config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json"]
missing = [n for n in need if not (p / n).exists()]
if missing:
    raise SystemExit("missing files: " + str(missing))
size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
print(f"[OK] {p}  {size/1e9:.2f} GB")
for n in need:
    print(" ", n, (p / n).stat().st_size)
PY
