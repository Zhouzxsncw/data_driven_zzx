#!/usr/bin/env bash
# 江算 π₀.₅：自动加载 cigia_lerobot 源码环境，再按 YAML 训练。
# 数据集路径读 YAML 里的 data.repo_id（与 cigia_lerobot 无关）。
# 用法:
#   bash scripts/train_pi05_ppu_yaml.sh
#   bash scripts/train_pi05_ppu_yaml.sh --dry-run
#   bash scripts/train_pi05_ppu_yaml.sh --config configs/train_pi05.yaml
# fsdp_devices>1 时自动 torchrun；单卡仍是一条 python 进程。
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/env_ppu.sh"

CONFIG="${REPO_DIR}/configs/train_pi05.yaml"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${NPROC:-}" ]]; then
  NPROC="$("${PYTHON_BIN}" -c "import yaml,sys; print(int((yaml.safe_load(open(sys.argv[1])) or {}).get('fsdp_devices', 1) or 1))" "${CONFIG}" 2>/dev/null || echo 1)"
fi
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

DRY=0
for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
  [[ "$a" == "--dry-run" ]] && DRY=1
done

if [[ "${NPROC}" -gt 1 && "${DRY}" -eq 0 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  exec "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-29701}" \
    "${REPO_DIR}/scripts/train_from_yaml.py" --config "${CONFIG}" "${EXTRA_ARGS[@]}"
else
  exec "${PYTHON_BIN}" "${REPO_DIR}/scripts/train_from_yaml.py" --config "${CONFIG}" "${EXTRA_ARGS[@]}"
fi
