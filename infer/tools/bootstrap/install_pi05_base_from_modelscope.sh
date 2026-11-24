#!/usr/bin/env bash
# [维护者] 将 ModelScope / 本机缓存中的 pi05_base（约 14GB）链入 data/hf_hub。
# 实习生训练请直接使用 data/hf_hub 内已有 snapshot，勿运行本脚本。
#
# 用法（仓库根）:
#   bash tools/bootstrap/install_pi05_base_from_modelscope.sh
#   MS_PI05_DIR=/path/to/pi05_base bash tools/bootstrap/install_pi05_base_from_modelscope.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
cd "$ROOT"
# shellcheck source=../train/scripts/lib/sim_offline_env.sh
source "${ROOT}/train/scripts/lib/sim_offline_env.sh"
sim_apply_offline_env

MS_DIR="${MS_PI05_DIR:-${HOME}/.cache/modelscope/hub/models/lerobot/pi05_base}"
HF_REPO="${HF_HOME}/hub/models--lerobot--pi05_base"
SNAP="${PI05_HF_SNAPSHOT:-modelscope-main}"

if [[ ! -f "${MS_DIR}/model.safetensors" ]]; then
  echo "错误: 未找到 ${MS_DIR}/model.safetensors" >&2
  echo "  请先在 ModelScope 下载 lerobot/pi05_base，或设置 MS_PI05_DIR" >&2
  exit 1
fi

echo "源: ${MS_DIR}"
echo "目标: ${HF_REPO}/snapshots/${SNAP}"
mkdir -p "${HF_REPO}/snapshots" "${HF_REPO}/refs"

# 清理此前中断的 HF 下载残留
find "${HF_REPO}" -name '*.incomplete' -delete 2>/dev/null || true

echo "拷贝中（约 14GB，请耐心等待）..."
rsync -a --info=progress2 \
  --exclude='.mdl' --exclude='.msc' --exclude='.mv' --exclude='.cache' \
  "${MS_DIR}/" "${HF_REPO}/snapshots/${SNAP}/"

printf '%s' "${SNAP}" >"${HF_REPO}/refs/main"

PY="${LEROBOT_PYTHON:-${ROOT}/.venv310/bin/python}"
export PYTHONPATH="${ROOT}/train/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
import os
from pathlib import Path
from transformers.utils import cached_file

hf_home = Path(os.environ["HF_HOME"])
resolved = cached_file(
    "lerobot/pi05_base",
    "model.safetensors",
    local_files_only=True,
)
size_gb = Path(resolved).stat().st_size / (1024**3)
print(f"OK: model.safetensors -> {resolved} ({size_gb:.2f} GB)")
PY

echo "完成。训练将使用本地 snapshot: ${HF_HOME}/hub/models--lerobot--pi05_base/snapshots/*（见 train_pi05.sh）"
