#!/usr/bin/env bash
# 在 h100-103 把 00091 关节 v2.1 转成 τ0-WM EE v3.0（列约定 rot6d）。
# 不占 GPU，可与 00073 训练并行。
#
#   setsid nohup bash /data1/tau0wm/local_scripts/h100/convert_00091_ee_on_h100.sh \
#       > /data1/tau0wm/convert_00091.out 2>&1 &
set -uo pipefail

ROOT="${TAU_H100_ROOT:-/data1/tau0wm}"
S="${ROOT}/local_scripts"
VENV="${ROOT}/envs/tau0wm"
DATASETS="${ROOT}/datasets"
OUT="${DATASETS}/00091_rm75_ee_tau0wm"
SRC="${SRC:-/data1/dataset/00091_20260903_RM_01_yj_21}"
LOG="${ROOT}/convert_00091.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

[[ -x "${VENV}/bin/python" ]] || die "venv 不在"
[[ -f "${S}/convert_00036_ee.py" ]] || die "缺 convert_00036_ee.py"
[[ -f "${S}/convert_tongbot_joint_to_ee.py" ]] || die "缺 convert_tongbot_joint_to_ee.py"
[[ -f "${S}/RM_API/Robotic_Arm/libs/linux_x86/libapi_c.so" ]] || die "缺 RM_API"
[[ -f "${SRC}/meta/info.json" ]] || die "缺源数据集 ${SRC}"

if [[ -f "${OUT}/meta/info.json" ]] \
   && grep -q '"codebase_version": "v3.0"' "${OUT}/meta/info.json" \
   && [[ -f "${S}/configs/stat_file/tongbot_00091_ee_rela.json" ]]; then
  log "已是 v3.0 + 统计，跳过"
  exit 0
fi
: > "${LOG}"

"${VENV}/bin/python" - "${SRC}" <<'PY' 2>&1 | tee -a "${LOG}" || die "源数据集规模不符"
import json, sys
m = json.load(open(f"{sys.argv[1]}/meta/info.json"))
print("codebase", m["codebase_version"], "fps", m["fps"],
      "ep", m["total_episodes"], "frames", m["total_frames"])
assert m["codebase_version"] == "v2.1", m["codebase_version"]
assert m["total_episodes"] == 500 and m["total_frames"] == 208348, m
assert m["fps"] == 15
PY

log "[1/3] joint→EE（RM75 离线 FK，J7 +120°，列约定 rot6d）"
rm -rf "${OUT}" "${OUT}_old"
mkdir -p "${DATASETS}"
TAU_CONVERTER_DIR="${S}" RM_API="${S}/RM_API" \
  PYTHONPATH="${S}" "${VENV}/bin/python" "${S}/convert_00036_ee.py" \
    --src "${SRC}" --dst "${OUT}" \
    --sdk-root "${S}/RM_API" --force-type RM_MODEL_RM_B_E --j7-offset-deg 120 \
    --overwrite 2>&1 | tee -a "${LOG}" || die "FK 失败"

log "[2/3] v2.1 → v3.0"
( cd "${ROOT}" && "${VENV}/bin/python" -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id 00091_rm75_ee_tau0wm --root "${DATASETS}" --push-to-hub=false ) \
    >>"${LOG}" 2>&1 || { tail -40 "${LOG}"; die "v21→v30 失败，见 ${LOG}"; }

log "[3/3] 算 EE 20D 相对统计"
cp -f "${S}/configs/data/tongbot_00091_ee_rela.yaml" \
      "${ROOT}/repo/tau-0-wm/configs/data/tongbot/00091_ee_rela.yaml"
( cd "${ROOT}/repo/tau-0-wm" && PYTHONPATH=. "${VENV}/bin/python" scripts/compute_stat.py \
    -c configs/data/tongbot/00091_ee_rela.yaml \
    --action_space eef6d --action_type relative \
    -o /tmp/stat_00091.json --num 200000 --batch_size 64 --num_workers 16 ) >>"${LOG}" 2>&1 \
  || die "compute_stat 失败"
mkdir -p "${S}/configs/stat_file" "${ROOT}/repo/tau-0-wm/configs/data/stat_file"
cp -f /tmp/stat_00091.json "${S}/configs/stat_file/tongbot_00091_ee_rela.json"
cp -f /tmp/stat_00091.json "${ROOT}/repo/tau-0-wm/configs/data/stat_file/tongbot_00091_ee_rela.json"
"${VENV}/bin/python" - /tmp/stat_00091.json <<'PY' 2>&1 | tee -a "${LOG}"
import json, sys
import numpy as np
a = json.load(open(sys.argv[1]))
for k in ("action", "state"):
    mean, std = np.array(a[k]["mean"]), np.array(a[k]["std"])
    assert mean.shape == std.shape == (20,), (k, mean.shape)
    print(f"{k}: mean[:4]={mean[:4].round(4).tolist()} std_min={std.min():.4f}")
    assert np.all(std > 1e-6), k
print("STAT_OK")
PY

du -sh "${OUT}" | tee -a "${LOG}"
log "EE_CONVERT_DONE"
