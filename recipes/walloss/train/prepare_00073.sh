#!/bin/bash
# 开训前准备：转 norm_stats、装 mistral-common、下 Qwen processor、探数据。
set -euo pipefail
ROOT="${ROOT:-/workspace}"
VENV="${VENV:-${ROOT}/venv}"
DATA="${DATA:-/data/datasets/00073_20260818_RM_01_yj_21}"
DST_NORM="${DST_NORM:-${ROOT}/outputs/norm_stats_00073_wallx.json}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

set +u
source /usr/local/PPU_SDK/envsetup.sh
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
set -u

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT}/hf}"
export PYTHONUNBUFFERED=1
mkdir -p "${ROOT}/outputs" "${ROOT}/logs"

echo "[1/5] convert norm_stats"
SRC_NORM=""
for cand in \
  "${DATA}/meta/norm_stats.json" \
  "${DATA}/norm_stats.json" \
  "${DATA}/meta/stats.json"; do
  if [ -f "${cand}" ]; then
    SRC_NORM="${cand}"
    break
  fi
done
if [ -z "${SRC_NORM}" ]; then
  echo "[ERR] no norm_stats.json under ${DATA}"
  find "${DATA}" -maxdepth 2 -name '*norm*' -o -name '*stat*' | head
  exit 1
fi
python /workspace/code/train/convert_norm_stats.py "${SRC_NORM}" "${DST_NORM}"

echo "[2/5] mistral-common (transformers 5.2 AutoProcessor)"
python -m pip install -q -U "mistral-common>=1.8.0" -i "${PIP_INDEX}" || \
  python -m pip install -q -U mistral-common -i "${PIP_INDEX}"

echo "[3/5] Qwen processor"
bash /workspace/code/train/download_qwen_processor.sh

echo "[4/5] AutoProcessor"
python - << PY
from transformers import AutoProcessor
from pathlib import Path
cands = [
    Path("/workspace/pretrained/Qwen2.5-VL-3B-Instruct"),
    Path("/workspace/pretrained/wall-oss-0.5"),
]
ok = None
for p in cands:
    if not (p / "tokenizer.json").exists() and not (p / "tokenizer_config.json").exists():
        print("[skip]", p)
        continue
    try:
        proc = AutoProcessor.from_pretrained(str(p), use_fast=True, trust_remote_code=True)
        print("[OK] AutoProcessor", p, type(proc).__name__, "tok", type(proc.tokenizer).__name__)
        ok = p
        break
    except Exception as e:
        print("[FAIL]", p, type(e).__name__, e)
if ok is None:
    raise SystemExit("AutoProcessor failed on all candidates")
PY

echo "[5/6] convert 00073 v2.1 -> v3"
bash /workspace/code/train/convert_00073_v3.sh

echo "[6/6] LeRobot 00073 probe"
python /workspace/code/train/probe_00073.py
echo "[OK] prepare done"
