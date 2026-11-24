#!/bin/bash
# 把 00073 从 LeRobot v2.1 转成 v3，写到 /workspace/datasets（不碰 openpi 原盘）。
set -euo pipefail
ROOT="${ROOT:-/workspace}"
VENV="${VENV:-${ROOT}/venv}"
SRC="${SRC:-/data/datasets/00073_20260818_RM_01_yj_21}"
DST="${DST:-${ROOT}/datasets/00073_20260818_RM_01_yj_21}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

set +u
source /usr/local/PPU_SDK/envsetup.sh
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
set -u

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

echo "[1/4] lerobot extras (no torch/numpy upgrade)"
python -m pip install -q \
  "draccus>=0.11.6,<0.12.0" \
  "jsonlines>=4.0.0,<5.0.0" \
  "av>=12,<16" \
  "pandas>=2.0,<3" \
  "gymnasium>=1.1.1,<2.0.0" \
  "termcolor>=2.4.0,<4.0.0" \
  -i "${PIP_INDEX}"

if [ ! -f "${DST}/meta/info.json" ]; then
  echo "[2/4] copy v2.1 ${SRC} -> ${DST}"
  mkdir -p "$(dirname "${DST}")"
  rsync -a --info=progress2 "${SRC}/" "${DST}/"
else
  echo "[2/4] dst already present: ${DST}"
fi

python - << PY
import json
from pathlib import Path
info = json.loads(Path("${DST}/meta/info.json").read_text())
print("current codebase_version", info.get("codebase_version"))
PY

ver="$(python -c 'import json; print(json.load(open("'"${DST}"'/meta/info.json")).get("codebase_version",""))')"
if [ "${ver}" = "v3.0" ]; then
  echo "[3/4] already v3.0, skip convert"
else
  echo "[3/4] convert v2.1 -> v3.0 (in ${DST}, original openpi copy untouched)"
  python /workspace/src/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py \
    --repo-id local/00073 \
    --root "${DST}" \
    --push-to-hub=false
fi

echo "[4/4] cleanup sibling _old if convert succeeded"
if [ -d "${DST}_old" ]; then
  python - << PY
import json
from pathlib import Path
p = Path("${DST}/meta/info.json")
ver = json.loads(p.read_text()).get("codebase_version")
assert ver == "v3.0", ver
print("v3 ok, removing ${DST}_old")
PY
  rm -rf "${DST}_old"
fi

python - << PY
import json
from pathlib import Path
info = json.loads(Path("${DST}/meta/info.json").read_text())
print("[OK] converted", info.get("codebase_version"),
      "episodes", info.get("total_episodes"),
      "frames", info.get("total_frames"))
PY
