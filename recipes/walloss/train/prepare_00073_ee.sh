#!/bin/bash
# 00073: 16D joint → 绝对 EE → LeRobot v3，并按 relative EE 算 wall-x norm。
# 不碰 openpi 原盘。容器内执行。
set -euo pipefail
ROOT="${ROOT:-/workspace}"
VENV="${VENV:-${ROOT}/venv}"
SRC_V21="${SRC_V21:-/data/datasets/00073_20260818_RM_01_yj_21}"
DST_EE="${DST_EE:-${ROOT}/datasets/00073_rm75_ee}"
SDK="${SDK:-${ROOT}/sdk/RM_API}"
CFG="${CFG:-${ROOT}/code/train/sft_tongbot_ee_00073.yml}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

set +u
source /usr/local/PPU_SDK/envsetup.sh
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
set -u
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

test -f "${SDK}/Robotic_Arm/rm_robot_interface.py"
test -f "${SRC_V21}/meta/info.json"

echo "[1/4] FK joint → absolute EE (v2.1 parquet)"
python /workspace/code/scripts/convert_tongbot_joint_to_ee.py \
  --src "${SRC_V21}" \
  --dst "${DST_EE}" \
  --sdk-root "${SDK}" \
  --overwrite

echo "[2/4] v2.1 → v3.0 (current lerobot)"
python - << PY
import json
from pathlib import Path
info = json.loads(Path("${DST_EE}/meta/info.json").read_text())
print("before convert", info.get("codebase_version"))
PY
ver="$(python -c 'import json; print(json.load(open("'"${DST_EE}"'/meta/info.json")).get("codebase_version",""))')"
if [ "${ver}" != "v3.0" ]; then
  python /workspace/src/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py \
    --repo-id local/00073_ee \
    --root "${DST_EE}" \
    --push-to-hub=false
  if [ -d "${DST_EE}_old" ]; then
    rm -rf "${DST_EE}_old"
  fi
fi

echo "[3/4] wall-x relative EE norm_stats"
mkdir -p "${ROOT}/outputs"
python /workspace/src/wall-x/scripts/compute_norm_stats.py \
  --train_config "${CFG}"

echo "[4/4] probe one EE sample"
python - << PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
repo = "${DST_EE}"
meta = LeRobotDatasetMetadata(repo, root=None)
print("v3", meta.total_episodes, "frames", meta.total_frames, "fps", meta.fps)
print("features", list(meta.features))
assert "observation.state.ee" in meta.features
assert "action.ee" in meta.features
ds = LeRobotDataset(repo, root=None, delta_timestamps={"action.ee": [t / float(meta.fps) for t in range(32)]})
s = ds[0]
print("action.ee", tuple(s["action.ee"].shape), "state.ee", tuple(s["observation.state.ee"].shape))
print("task", s.get("task", "")[:120])
assert s["action.ee"].shape[-1] == 20
assert s["observation.state.ee"].shape[-1] == 20
print("[OK] EE dataset ready")
PY
