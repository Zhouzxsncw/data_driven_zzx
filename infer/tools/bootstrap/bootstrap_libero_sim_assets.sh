#!/usr/bin/env bash
# LIBERO 仿真资源 → data/libero_assets/（训中 eval，与 LeRobot 训练集无关）
#
# 默认：仅校验仓库内 data/libero_assets/，不联网。
# 维护者联网填充: ALLOW_LIBERO_ASSETS_DOWNLOAD=1 bash tools/bootstrap/bootstrap_libero_sim_assets.sh
set -euo pipefail

PY="${LEROBOT_PYTHON:-python3}"

ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
if [[ -f "${ROOT}/train/scripts/lib/sim_offline_env.sh" ]]; then
  # shellcheck source=../train/scripts/lib/sim_offline_env.sh
  source "${ROOT}/train/scripts/lib/sim_offline_env.sh"
  sim_apply_offline_env
fi

ASSETS_DIR="${LIBERO_ASSETS_DIR:-${ROOT}/data/libero_assets}"
CFG_DIR="${LIBERO_CONFIG_PATH:-${ROOT}/data/libero_config}"
mkdir -p "$ASSETS_DIR" "$CFG_DIR"

if [[ -d "${ASSETS_DIR}/scenes" && -d "${ASSETS_DIR}/stable_scanned_objects" ]]; then
  echo "[bootstrap] LIBERO assets 已存在: ${ASSETS_DIR}"
else
  MS_ASSETS="${HOME}/.cache/modelscope/hub/datasets/lerobot/libero-assets"
  if [[ "${ALLOW_LIBERO_ASSETS_DOWNLOAD:-}" == "1" ]]; then
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
    if [[ -d "${MS_ASSETS}/scenes" ]]; then
      echo "[bootstrap] 使用 ModelScope 缓存: ${MS_ASSETS}"
      rm -rf "${ASSETS_DIR}"
      ln -sfn "$(readlink -f "${MS_ASSETS}")" "${ASSETS_DIR}"
    else
      echo "[bootstrap] 从 Hub 下载 lerobot/libero-assets -> ${ASSETS_DIR}"
      "$PY" - <<PY
from huggingface_hub import snapshot_download
import os
assets = os.path.expanduser("${ASSETS_DIR}")
snapshot_download(
    "lerobot/libero-assets",
    repo_type="dataset",
    local_dir=assets,
)
print("done", assets)
PY
    fi
  else
    echo "错误: 未找到本地 LIBERO assets: ${ASSETS_DIR}" >&2
    echo "  实习生: 使用含 data/libero_assets/ 的完整仓库（见 data/README.md）" >&2
    echo "  维护者: ALLOW_LIBERO_ASSETS_DOWNLOAD=1 bash $0" >&2
    exit 1
  fi
fi

PKG="$("$PY" -c "import libero, os; print(os.path.dirname(libero.__file__))")"
cat > "${CFG_DIR}/config.yaml" <<EOF
benchmark_root: ${PKG}/libero
bddl_files: ${PKG}/libero/bddl_files
init_states: ${PKG}/libero/init_files
datasets: ${PKG}/../datasets
assets: ${ASSETS_DIR}
EOF
echo "[bootstrap] LIBERO config: ${CFG_DIR}/config.yaml (assets=${ASSETS_DIR})"

# libero.get_assets_path() 优先读包内 ./assets，须链到 data/libero_assets
LIBERO_PKG_ASSETS="${PKG}/libero/assets"
ln -sfn "${ASSETS_DIR}" "${LIBERO_PKG_ASSETS}"
echo "[bootstrap] symlink: ${LIBERO_PKG_ASSETS} -> ${ASSETS_DIR}"
