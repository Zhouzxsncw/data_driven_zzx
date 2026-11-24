#!/usr/bin/env bash
# 把 recipes/tau0_wm/configs 下的配置同步进 τ0-WM 仓库（训练时按仓库相对路径读取）。
# 约定：本 recipe 目录是配置的**唯一真源**，repo 里的那份只是副本，别在 repo 里直接改。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAU_ROOT="${TAU_ROOT:-/home/xk/VLA/tau-0-wm}"

[[ -d "${TAU_ROOT}/configs" ]] || { echo "找不到 τ0-WM 仓库：${TAU_ROOT}" >&2; exit 1; }

mkdir -p "${HERE}/configs/stat_file" "${TAU_ROOT}/configs/data/tongbot" "${TAU_ROOT}/configs/tau_model" \
         "${TAU_ROOT}/configs/data/stat_file"

cp -v "${HERE}/configs/data/tongbot_00036_abs_joint.yaml" \
      "${TAU_ROOT}/configs/data/tongbot/00036_abs_joint.yaml"
cp -v "${HERE}/configs/data/tongbot_00036_ee_rela.yaml" \
      "${TAU_ROOT}/configs/data/tongbot/00036_ee_rela.yaml"
cp -v "${HERE}/configs/tau_model/posttrain_tongbot_00036.yaml" \
      "${TAU_ROOT}/configs/tau_model/posttrain_tongbot_00036.yaml"
cp -v "${HERE}/configs/tau_model/posttrain_tongbot_00036_ee.yaml" \
      "${TAU_ROOT}/configs/tau_model/posttrain_tongbot_00036_ee.yaml"
for pair in \
  "data/tongbot_00073_ee_rela.yaml:data/tongbot/00073_ee_rela.yaml" \
  "data/tongbot_00091_ee_rela.yaml:data/tongbot/00091_ee_rela.yaml" \
  "data/tongbot_00108_ee_rela.yaml:data/tongbot/00108_ee_rela.yaml" \
  "tau_model/posttrain_tongbot_00073_ee_300k.yaml:tau_model/posttrain_tongbot_00073_ee_300k.yaml" \
  "tau_model/posttrain_tongbot_00091_ee_300k.yaml:tau_model/posttrain_tongbot_00091_ee_300k.yaml"
do
  src="${HERE}/configs/${pair%%:*}"
  dst="${TAU_ROOT}/configs/${pair##*:}"
  [[ -f "${src}" ]] && cp -v "${src}" "${dst}"
done

# stat_file：recipe 里有就以 recipe 为准（推给 h100 时靠这一份），
# 否则把 repo 里新生成的回流到 recipe 留档（compute_stat 是在 repo 里跑的）
for s in tongbot_00036_abs tongbot_00036_ee_rela tongbot_00073_ee_rela tongbot_00091_ee_rela tongbot_00108_ee_rela; do
  if [[ -f "${HERE}/configs/stat_file/${s}.json" ]]; then
    cp -v "${HERE}/configs/stat_file/${s}.json" \
          "${TAU_ROOT}/configs/data/stat_file/${s}.json"
  elif [[ -f "${TAU_ROOT}/configs/data/stat_file/${s}.json" ]]; then
    cp -v "${TAU_ROOT}/configs/data/stat_file/${s}.json" \
          "${HERE}/configs/stat_file/${s}.json"
  fi
done

echo "[tau0_wm] 同步完成 → ${TAU_ROOT}/configs"
