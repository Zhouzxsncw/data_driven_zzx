#!/usr/bin/env bash
# 在 h100-103 把 00108 关节 v2.1 转成 τ0-WM EE v3.0（列约定 rot6d）。
# 源已在 /data1/dataset，不占 GPU，可与 00091 训练并行。
# depth PNG 不进 EE 树（τ0-WM 只用三路 RGB）。
#
#   setsid nohup bash /data1/tau0wm/local_scripts/h100/convert_00108_ee_on_h100.sh \
#       > /data1/tau0wm/convert_00108.out 2>&1 &
set -uo pipefail

ROOT="${TAU_H100_ROOT:-/data1/tau0wm}"
S="${ROOT}/local_scripts"
VENV="${ROOT}/envs/tau0wm"
DATASETS="${ROOT}/datasets"
OUT="${DATASETS}/00108_rm75_ee_tau0wm"
SRC="${SRC:-/data1/dataset/00108_20260922_RM_01_YJ_v21}"
LOG="${ROOT}/convert_00108.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

[[ -x "${VENV}/bin/python" ]] || die "venv 不在"
[[ -f "${S}/convert_00036_ee.py" ]] || die "缺 convert_00036_ee.py"
[[ -f "${S}/convert_tongbot_joint_to_ee.py" ]] || die "缺 convert_tongbot_joint_to_ee.py"
[[ -f "${S}/RM_API/Robotic_Arm/libs/linux_x86/libapi_c.so" ]] || die "缺 RM_API"
[[ -f "${SRC}/meta/info.json" ]] || die "缺源数据集 ${SRC}"

if [[ -f "${OUT}/meta/info.json" ]] \
   && grep -q '"codebase_version": "v3.0"' "${OUT}/meta/info.json" \
   && [[ -f "${S}/configs/stat_file/tongbot_00108_ee_rela.json" ]]; then
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
assert m["total_episodes"] == 200 and m["total_frames"] == 24290, m
assert m["fps"] == 15
cams = {k for k in m.get("features", {}) if k.startswith("observation.images.")}
assert cams == {
    "observation.images.cam_left",
    "observation.images.cam_mid",
    "observation.images.cam_right",
}, cams
PY

log "[1/4] joint→EE（RM75 离线 FK，J7 +120°，列约定 rot6d）"
rm -rf "${OUT}" "${OUT}_old"
mkdir -p "${DATASETS}"
TAU_CONVERTER_DIR="${S}" RM_API="${S}/RM_API" \
  PYTHONPATH="${S}" "${VENV}/bin/python" "${S}/convert_00036_ee.py" \
    --src "${SRC}" --dst "${OUT}" \
    --sdk-root "${S}/RM_API" --force-type RM_MODEL_RM_B_E --j7-offset-deg 120 \
    --overwrite 2>&1 | tee -a "${LOG}" || die "FK 失败"

log "[2/4] 补 v2.1 必填字段 + 去掉 depth（不拷 7 万张 PNG）"
"${VENV}/bin/python" - "${OUT}/meta/info.json" <<'PY' 2>&1 | tee -a "${LOG}" || die "补 info.json 失败"
import json, sys
p = sys.argv[1]
info = json.load(open(p))
feats = info.setdefault("features", {})
drop = [k for k in list(feats) if k.startswith("observation.depths.")]
for k in drop:
    del feats[k]
# 00108 采集树没有 total_chunks / total_videos，lerobot 0.4.4 的 v21→v30 会 del 这两个键
n_ep = int(info["total_episodes"])
chunk = int(info.get("chunks_size") or 1000)
n_vid = sum(1 for v in feats.values() if isinstance(v, dict) and v.get("dtype") == "video")
info["total_chunks"] = max(1, (n_ep + chunk - 1) // chunk)
info["total_videos"] = n_ep * n_vid
json.dump(info, open(p, "w"), indent=4, ensure_ascii=False)
print("dropped", drop, "total_chunks", info["total_chunks"], "total_videos", info["total_videos"])
PY

log "[3/4] v2.1 → v3.0"
( cd "${ROOT}" && "${VENV}/bin/python" -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id 00108_rm75_ee_tau0wm --root "${DATASETS}" --push-to-hub=false ) \
    >>"${LOG}" 2>&1 || { tail -40 "${LOG}"; die "v21→v30 失败，见 ${LOG}"; }

log "[4/4] 算 EE 20D 相对统计"
cp -f "${S}/configs/data/tongbot_00108_ee_rela.yaml" \
      "${ROOT}/repo/tau-0-wm/configs/data/tongbot/00108_ee_rela.yaml"
( cd "${ROOT}/repo/tau-0-wm" && PYTHONPATH=. "${VENV}/bin/python" scripts/compute_stat.py \
    -c configs/data/tongbot/00108_ee_rela.yaml \
    --action_space eef6d --action_type relative \
    -o /tmp/stat_00108.json --num 24290 --batch_size 64 --num_workers 8 ) >>"${LOG}" 2>&1 \
  || die "compute_stat 失败"
mkdir -p "${S}/configs/stat_file" "${ROOT}/repo/tau-0-wm/configs/data/stat_file"
cp -f /tmp/stat_00108.json "${S}/configs/stat_file/tongbot_00108_ee_rela.json"
cp -f /tmp/stat_00108.json "${ROOT}/repo/tau-0-wm/configs/data/stat_file/tongbot_00108_ee_rela.json"
"${VENV}/bin/python" - /tmp/stat_00108.json <<'PY' 2>&1 | tee -a "${LOG}"
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
