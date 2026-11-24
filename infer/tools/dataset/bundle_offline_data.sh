#!/usr/bin/env bash
# [维护者] 将本机已下载的 LeRobot 数据集、LIBERO 资源、HF 模型缓存复制到仓库 data/，
# 整仓 rsync 给实习生后，训练/评估/预训练均只读 data/，默认不联网。
#
# 用法（仓库根）:
#   bash tools/dataset/bundle_offline_data.sh              # 复制已有缓存（跳过已完整项）
#   bash tools/dataset/bundle_offline_data.sh --force      # 强制覆盖
#   bash tools/dataset/bundle_offline_data.sh --with-metaworld   # 含 metaworld_mt50（约 40G+）
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
cd "$ROOT"
# shellcheck source=../train/scripts/lib/sim_offline_env.sh
source "${ROOT}/train/scripts/lib/sim_offline_env.sh"
sim_apply_offline_env

FORCE=0
WITH_METAWORLD=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --with-metaworld) WITH_METAWORLD=1 ;;
  esac
done

RSYNC=(rsync -a --info=progress2)
[[ "$FORCE" -eq 1 ]] && RSYNC+=(--delete)

DEST_LEROBOT="${OFFLINE_DATA_ROOT}/lerobot"
DEST_HF="${HF_HOME}/hub"
DEST_ASSETS="${LIBERO_ASSETS_DIR}"
mkdir -p "$DEST_LEROBOT" "$DEST_HF" "$DEST_ASSETS" "${LIBERO_CONFIG_PATH}"

dataset_complete() {
  [[ -f "$1/meta/info.json" ]]
}

_copy_dataset() {
  local name=$1 src=$2
  local dest="${DEST_LEROBOT}/${name}"
  if dataset_complete "$dest" && [[ "$FORCE" -eq 0 ]]; then
    echo "[SKIP] ${name} 已存在: ${dest}"
    return 0
  fi
  if ! dataset_complete "$src"; then
    echo "[SKIP] 源不完整，跳过 ${name}: ${src}" >&2
    return 1
  fi
  echo "[COPY] ${src} -> ${dest}"
  rm -rf "${dest}"
  mkdir -p "$(dirname "$dest")"
  "${RSYNC[@]}" "${src}/" "${dest}/"
  echo "[OK] ${name}"
}

# --- LeRobot 数据集（优先 ModelScope，其次 HF lerobot 缓存）---
MS="${HOME}/.cache/modelscope/hub/datasets/lerobot"
HF_LR="${HOME}/.cache/huggingface/lerobot/lerobot"

for name in \
  aloha_sim_insertion_human aloha_sim_insertion_scripted \
  aloha_sim_transfer_cube_human aloha_sim_transfer_cube_scripted \
  aloha_sim_insertion_human_image aloha_sim_insertion_scripted_image \
  aloha_sim_transfer_cube_human_image aloha_sim_transfer_cube_scripted_image \
  libero libero_10 libero_spatial libero_object libero_goal libero_90 pusht; do
  src=""
  if dataset_complete "${MS}/${name}"; then
    src="${MS}/${name}"
  elif dataset_complete "${HF_LR}/${name}"; then
    src="${HF_LR}/${name}"
  elif dataset_complete "${ROOT}/data/lerobot/${name}"; then
    src="${ROOT}/data/lerobot/${name}"
  fi
  [[ -n "$src" ]] && _copy_dataset "$name" "$src" || true
done

# LIBERO 套件 episode 索引（合并集子集训练）
if [[ -d "${ROOT}/data/lerobot/_suite_indices" ]]; then
  echo "[COPY] _suite_indices -> ${DEST_LEROBOT}/_suite_indices"
  mkdir -p "${DEST_LEROBOT}/_suite_indices"
  "${RSYNC[@]}" "${ROOT}/data/lerobot/_suite_indices/" "${DEST_LEROBOT}/_suite_indices/"
fi

if [[ "$WITH_METAWORLD" -eq 1 ]]; then
  if dataset_complete "${MS}/metaworld_mt50"; then
    _copy_dataset metaworld_mt50 "${MS}/metaworld_mt50"
  elif dataset_complete "${HF_LR}/metaworld_mt50"; then
    _copy_dataset metaworld_mt50 "${HF_LR}/metaworld_mt50"
  else
    echo "[WARN] 未找到完整 metaworld_mt50，跳过（可加 --with-metaworld 前先下载）" >&2
  fi
fi

# 训练默认 repo_id=lerobot/metaworld → 目录名 metaworld
if dataset_complete "${DEST_LEROBOT}/metaworld_mt50" && ! dataset_complete "${DEST_LEROBOT}/metaworld"; then
  echo "[LINK] metaworld -> metaworld_mt50"
  ln -sfn "metaworld_mt50" "${DEST_LEROBOT}/metaworld"
fi

# --- LIBERO 仿真 assets ---
_ms_assets="${MS}/libero-assets"
if [[ -d "${_ms_assets}/scenes" ]]; then
  echo "[COPY] libero-assets -> ${DEST_ASSETS}"
  "${RSYNC[@]}" "${_ms_assets}/" "${DEST_ASSETS}/"
elif [[ -d "${HOME}/.cache/libero/assets/scenes" ]]; then
  "${RSYNC[@]}" "${HOME}/.cache/libero/assets/" "${DEST_ASSETS}/"
fi

# --- HF Hub 模型（PI0.5 / paligemma）---
_copy_hf_repo() {
  local hub_name=$1
  local src="${HOME}/.cache/huggingface/hub/${hub_name}"
  local dest="${DEST_HF}/${hub_name}"
  if [[ ! -d "$src" ]]; then
    echo "[SKIP] 无 HF 缓存: ${hub_name}" >&2
    return 1
  fi
  if [[ -d "${dest}/snapshots" ]] && [[ "$FORCE" -eq 0 ]]; then
    echo "[SKIP] ${hub_name}"
    return 0
  fi
  echo "[COPY] ${hub_name} -> ${dest}"
  mkdir -p "$(dirname "$dest")"
  "${RSYNC[@]}" "${src}/" "${dest}/"
}

_copy_hf_repo "models--google--paligemma-3b-pt-224" || true
_copy_hf_repo "models--lerobot--pi05_base" || true
_copy_hf_repo "models--nvidia--GR00T-N1.5-3B" || true
_copy_hf_repo "models--lerobot--eagle2hg-processor-groot-n1p5" || true

# --- 写入仓库内 LIBERO 配置 ---
PY="${LEROBOT_PYTHON:-${ROOT}/.venv310/bin/python}"
if [[ -x "$PY" ]]; then
  pkg="$("$PY" -c "import libero, os; print(os.path.dirname(libero.__file__))")"
  cat >"${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${pkg}/libero
bddl_files: ${pkg}/libero/bddl_files
init_states: ${pkg}/libero/init_files
datasets: ${pkg}/../datasets
assets: ${DEST_ASSETS}
EOF
  echo "[OK] LIBERO 配置: ${LIBERO_CONFIG_PATH}/config.yaml"
fi

# --- 清单 ---
MANIFEST="${OFFLINE_DATA_ROOT}/MANIFEST.txt"
{
  echo "# CQTongBot offline data bundle — $(date -Iseconds)"
  echo "OFFLINE_DATA_ROOT=${OFFLINE_DATA_ROOT}"
  echo "HF_HOME=${HF_HOME}"
  echo ""
  echo "## LeRobot datasets"
  for d in "${DEST_LEROBOT}"/*; do
    [[ -d "$d" ]] || continue
    n="$(basename "$d")"
    if dataset_complete "$d"; then
      echo "  OK  ${n}  $(du -shL "$d" 2>/dev/null | cut -f1)"
    else
      echo "  --  ${n}  (incomplete)"
    fi
  done
  echo ""
  echo "## HF hub models"
  for d in "${DEST_HF}"/models--*; do
    [[ -d "$d" ]] || continue
    echo "  $(basename "$d")  $(du -sh "$d" 2>/dev/null | cut -f1)"
  done
  echo ""
  echo "## LIBERO assets"
  if [[ -d "${DEST_ASSETS}/scenes" ]]; then
    echo "  OK  $(du -sh "${DEST_ASSETS}" 2>/dev/null | cut -f1)"
  else
    echo "  MISSING"
  fi
} | tee "$MANIFEST"

echo ""
echo "完成。请将整个仓库（含 data/）复制给实习生；训练脚本已默认使用 data/ 路径。"
echo "查看清单: ${MANIFEST}"
du -sh "${OFFLINE_DATA_ROOT}" 2>/dev/null || true
