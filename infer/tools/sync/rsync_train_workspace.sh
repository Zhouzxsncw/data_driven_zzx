#!/usr/bin/env bash
# 将可训练工作树同步到远端，保证布局完整（禁止只拷 train/）。
#
# 远端目标布局:
#   <remote_dir>/
#     train/          # 本仓库 train/
#     tools/          # 本仓库 tools/（含 bootstrap，训练入口强依赖）
#     data/
#       lerobot/
#       hf_hub/
#       libero_assets/
#       libero_config/
#
# 用法（在仓库根）:
#   bash tools/sync/rsync_train_workspace.sh h100-103 /home/zgyd/new_lerobot_train
#   bash tools/sync/rsync_train_workspace.sh h100-103 /home/zgyd/new_lerobot_train --dry-run
#   bash tools/sync/rsync_train_workspace.sh h100-103 /home/zgyd/new_lerobot_train \
#       --dataset /media/.../data_00017_20260709_h85_left_no_depth_m0710
#
# SSH 选项可用环境变量覆盖:
#   RSYNC_SSH='ssh -F ~/.ssh/config -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes'
set -euo pipefail

usage() {
  cat <<'EOF'
用法: bash tools/sync/rsync_train_workspace.sh <ssh-host> <remote_dir> [选项]

选项:
  --dry-run              只打印 rsync 计划，不实际传输
  --dataset DIR          额外同步本地 LeRobot 数据集到 remote_dir/data/lerobot/<名>
  --with-data-placeholders  仅创建远端 data/{lerobot,hf_hub,...} 空目录（默认开启）
  -h, --help             显示帮助

必传内容: train/ + tools/
勿只拷贝 train/，否则会 ModuleNotFoundError: tools（缺 tools/bootstrap）。
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${1:-}"
REMOTE_DIR="${2:-}"
DRY_RUN=false
DATASET_SRC=""
WITH_PLACEHOLDERS=true

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi
shift 2

while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --dataset)
      [[ $# -ge 2 ]] || { echo "错误: --dataset 缺少路径" >&2; exit 2; }
      DATASET_SRC="$2"
      shift 2
      ;;
    --with-data-placeholders) WITH_PLACEHOLDERS=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "错误: 未知参数 $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$HOST" && -n "$REMOTE_DIR" ]] || { usage >&2; exit 2; }
[[ -d "${REPO_ROOT}/train/scripts" ]] || { echo "错误: 找不到 ${REPO_ROOT}/train/scripts" >&2; exit 2; }
[[ -d "${REPO_ROOT}/tools/bootstrap" ]] || { echo "错误: 找不到 ${REPO_ROOT}/tools/bootstrap" >&2; exit 2; }

RSYNC_SSH="${RSYNC_SSH:-ssh -o BatchMode=yes}"
RSYNC_OPTS=(-az --delete --exclude '__pycache__/' --exclude '*.pyc' --exclude '.git/')
if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_OPTS+=(--dry-run --itemize-changes)
fi

echo "仓库根:   $REPO_ROOT"
echo "远端:     ${HOST}:${REMOTE_DIR}"
echo "同步:     train/ + tools/"
echo "=================================================="

# 确保远端父目录与标准布局
$RSYNC_SSH "$HOST" "mkdir -p $(printf '%q' "$REMOTE_DIR")/{train,tools,data/lerobot,data/hf_hub,data/libero_assets,data/libero_config}"

rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_SSH" \
  "${REPO_ROOT}/train/" \
  "${HOST}:${REMOTE_DIR}/train/"

rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_SSH" \
  "${REPO_ROOT}/tools/" \
  "${HOST}:${REMOTE_DIR}/tools/"

if [[ -n "$DATASET_SRC" ]]; then
  [[ -f "${DATASET_SRC}/meta/info.json" ]] || {
    echo "错误: 数据集缺少 meta/info.json: $DATASET_SRC" >&2
    exit 2
  }
  ds_name="$(basename "${DATASET_SRC%/}")"
  echo "同步数据集 -> data/lerobot/${ds_name}"
  $RSYNC_SSH "$HOST" "mkdir -p $(printf '%q' "${REMOTE_DIR}/data/lerobot/${ds_name}")"
  DS_RSYNC_OPTS=(-az --progress)
  if [[ "$DRY_RUN" == "true" ]]; then
    DS_RSYNC_OPTS+=(--dry-run)
  fi
  rsync "${DS_RSYNC_OPTS[@]}" -e "$RSYNC_SSH" \
    "${DATASET_SRC}/" \
    "${HOST}:${REMOTE_DIR}/data/lerobot/${ds_name}/"
fi

if [[ "$WITH_PLACEHOLDERS" == "true" ]]; then
  $RSYNC_SSH "$HOST" "mkdir -p $(printf '%q' "$REMOTE_DIR")/data/{lerobot,hf_hub,libero_assets,libero_config}"
fi

echo "=================================================="
echo "完成。远端应存在:"
echo "  ${REMOTE_DIR}/train/scripts/train.sh"
echo "  ${REMOTE_DIR}/tools/bootstrap/"
echo "  ${REMOTE_DIR}/data/lerobot/"
echo
echo "开训示例:"
echo "  ssh ${HOST} 'cd ${REMOTE_DIR} && bash train/scripts/train.sh --dry-run'"
echo "  # 若缺 accelerate: 在 train.yaml 设 training.python，或 export LEROBOT_PYTHON=..."
