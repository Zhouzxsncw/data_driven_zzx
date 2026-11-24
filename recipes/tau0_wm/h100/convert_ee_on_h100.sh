#!/usr/bin/env bash
# 在 h100-103 本地做 00036 的 joint→EE（而不是从 gpu-master 传 1.4 G 数据集过去）：
# 源 v2.1 数据本来就在 /data1/dataset，FK 只改 parquet、视频软链，产物 2 s 就出来，
# 省掉一次必然被网关掐断的大文件传输。
#
# 依赖 setup_h100.sh 已经把 venv（lerobot 0.4.4 / pandas / scipy）装好，
# 以及 push_to_h100.sh 已经把这三样传上来：
#   local_scripts/convert_00036_ee.py            （列约定 rot6d 的那份）
#   local_scripts/convert_tongbot_joint_to_ee.py （RealMan 离线 FK 本体）
#   local_scripts/RM_API/                        （Robotic_Arm SDK + linux_x86/libapi_c.so）
#
#   setsid nohup bash /data1/tau0wm/local_scripts/h100/convert_ee_on_h100.sh \
#       > /data1/tau0wm/convert_ee.out 2>&1 &
set -uo pipefail

ROOT="${TAU_H100_ROOT:-/data1/tau0wm}"
S="${ROOT}/local_scripts"
VENV="${ROOT}/envs/tau0wm"
DATASETS="${ROOT}/datasets"
OUT="${DATASETS}/00036_rm75_ee_tau0wm"
LOG="${ROOT}/convert_ee_h100.log"
: > "${LOG}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

[[ -x "${VENV}/bin/python" ]] || die "venv 不在（先跑 setup_h100.sh）"
[[ -f "${S}/convert_00036_ee.py" ]] || die "缺 ${S}/convert_00036_ee.py"
[[ -f "${S}/convert_tongbot_joint_to_ee.py" ]] || die "缺 ${S}/convert_tongbot_joint_to_ee.py"
[[ -f "${S}/RM_API/Robotic_Arm/libs/linux_x86/libapi_c.so" ]] || die "缺 RM_API"

# ---- SDK 能不能在这台机器上 dlopen + 正解返回有限值（glibc 版本坑） -----------
# 跨机器数值一致性由第 3 步的统计逐维比对兜底（50440 帧的 mean/std 对得上 = FK 一致）
"${VENV}/bin/python" - <<'PY' >>"${LOG}" 2>&1 || die "RM_API SDK 在 h100 上不可用，见 ${LOG}"
import sys
import numpy as np
sys.path.insert(0, "/data1/tau0wm/local_scripts/RM_API")
from Robotic_Arm.rm_robot_interface import Algo, rm_force_type_e, rm_robot_arm_model_e
algo = Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E, rm_force_type_e.RM_MODEL_RM_B_E)
assert int(algo.arm_dof) == 7, algo.arm_dof
pose = algo.rm_algo_forward_kinematics([0.0, -0.5, 0.5, 1.5, 0.0, 0.5, 0.0], 1)
print("RM_API OK, fk(零位附近) =", np.asarray(pose, dtype=float).round(4).tolist())
assert np.all(np.isfinite(np.asarray(pose, dtype=float))), pose
PY

# ---- 找源 v2.1 数据集（挑第一个确实是 v2.1 的 00036 目录） --------------------
find_src() {
  local d
  for d in $(ls -d /data1/dataset/00036* 2>/dev/null); do
    [[ -f "${d}/meta/info.json" ]] || continue
    if grep -q '"v2.1"' "${d}/meta/info.json"; then echo "${d}"; return 0; fi
  done
  return 1
}
SRC="${SRC:-$(find_src || true)}"
[[ -n "${SRC}" && -f "${SRC}/meta/info.json" ]] || die "找不到 00036 的 v2.1 源目录（用 SRC=/data1/dataset/xxx 指定）"
log "源 v2.1 = ${SRC}"
"${VENV}/bin/python" - "${SRC}" <<'PY' 2>&1 | tee -a "${LOG}" || die "源数据集版本/规模不符"
import json, sys
m = json.load(open(f"{sys.argv[1]}/meta/info.json"))
print("codebase", m["codebase_version"], "fps", m["fps"],
      "ep", m["total_episodes"], "frames", m["total_frames"])
assert m["codebase_version"] == "v2.1", m["codebase_version"]
assert m["total_episodes"] == 151 and m["total_frames"] == 50440, "与 gpu-master 上的 00036 不是同一份"
assert m["fps"] == 15
PY

# ---- 1) FK：joint 16D → 追加 20D 绝对 EE（列约定 rot6d） ----------------------
log "[1/3] joint→EE（RM75 离线 FK，J7 +120°）"
rm -rf "${OUT}" "${OUT}_old"
mkdir -p "${DATASETS}"
TAU_CONVERTER_DIR="${S}" RM_API="${S}/RM_API" \
  PYTHONPATH="${S}" "${VENV}/bin/python" "${S}/convert_00036_ee.py" \
    --src "${SRC}" --dst "${OUT}" \
    --sdk-root "${S}/RM_API" --force-type RM_MODEL_RM_B_E --j7-offset-deg 120 \
    --overwrite 2>&1 | tee -a "${LOG}" || die "FK 失败"

# ---- 2) v2.1 → v3.0（lerobot 0.4.4；视频 stream-copy 合并，不重编码） ---------
# 注意模块路径：0.4.4 在 lerobot.datasets.v30.*，不是老版本的 lerobot.scripts.*
# --root 是**父目录**，转换器自己去 root/repo-id 找 v2.1，产物落回同一路径、原件移到 _old
log "[2/3] v2.1 → v3.0"
( cd "${ROOT}" && "${VENV}/bin/python" -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id 00036_rm75_ee_tau0wm --root "${DATASETS}" --push-to-hub=false ) \
    >>"${LOG}" 2>&1 || { tail -30 "${LOG}"; die "v21→v30 失败，见 ${LOG}"; }

# ---- 3) 与 gpu-master 产物对齐：重算统计并逐维比对 ---------------------------
# 同一条 FK 链路在两台机器上跑，数值必须一致；不一致说明 SDK/浮点有差异，
# 那就回退用 push 过来的 gpu-master 版本（数据+统计成对换掉）。
log "[3/3] 重算 EE 20D 统计并与 gpu-master 版比对"
cp -f "${S}/configs/data/tongbot_00036_ee_rela.yaml" \
      "${ROOT}/repo/tau-0-wm/configs/data/tongbot/00036_ee_rela.yaml"
( cd "${ROOT}/repo/tau-0-wm" && PYTHONPATH=. "${VENV}/bin/python" scripts/compute_stat.py \
    -c configs/data/tongbot/00036_ee_rela.yaml \
    --action_space eef6d --action_type relative \
    -o /tmp/stat_h100.json --num 100000 --batch_size 64 --num_workers 16 ) >>"${LOG}" 2>&1 \
  || die "compute_stat 失败"
"${VENV}/bin/python" - /tmp/stat_h100.json "${S}/configs/stat_file/tongbot_00036_ee_rela.json" <<'PY' 2>&1 | tee -a "${LOG}"
import json, sys
import numpy as np
a, b = (json.load(open(p)) for p in sys.argv[1:3])
worst = 0.0
for k in ("action", "state"):
    for s in ("mean", "std"):
        x, y = np.array(a[k][s]), np.array(b[k][s])
        assert x.shape == y.shape == (20,), (k, s, x.shape, y.shape)
        d = float(np.abs(x - y).max()); worst = max(worst, d)
        print(f"{k}.{s}: max|Δ|={d:.3e}")
print("两台机器 FK/统计一致" if worst < 1e-4 else f"⚠ 差异 {worst:.3e}：改用 gpu-master 那份数据")
assert worst < 1e-4, "统计不一致，需人工判断用哪一份"
PY

du -sh "${OUT}" | tee -a "${LOG}"
log "EE_CONVERT_DONE"
