#!/usr/bin/env bash
# Cosmos3-Nano 推理环境（gpu-master / RTX PRO 6000 Blackwell）。
# 对照 H100 /data1/setup_cosmos3_h100.sh：venv + donor torch + 本地 diffusers / cosmos-framework。
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export GIT_LFS_SKIP_SMUDGE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="${VLA_ROOT:-/home/xk/VLA}"
ENV_DIR="${ENV_DIR:-/data/cosmos3-infer}"
DONOR_ENV="${DONOR_ENV:-/home/xk/miniconda3/envs/lingbotva}"
DONOR_PY="${DONOR_PY:-${DONOR_ENV}/bin/python}"
COSMOS_DIR="${COSMOS_DIR:-${VLA_ROOT}/cosmos}"
FRAMEWORK_DIR="${FRAMEWORK_DIR:-${VLA_ROOT}/cosmos-framework}"
DIFFUSERS_DIR="${DIFFUSERS_DIR:-${VLA_ROOT}/cosmos_h100_setup/diffusers}"
MODEL_DIR="${MODEL_DIR:-${VLA_ROOT}/cosmos_h100_setup/Cosmos3-Nano}"
LOG_FILE="${LOG_FILE:-${ROOT}/setup_env.log}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

mkdir -p "$(dirname "$LOG_FILE")" "$ENV_DIR"

log "========== Cosmos3 gpu-master setup =========="
log "ENV_DIR=$ENV_DIR DONOR=$DONOR_ENV MODEL=$MODEL_DIR"

if [ ! -x "$DONOR_PY" ]; then
  echo "donor python 不存在: $DONOR_PY" >&2
  exit 1
fi
if [ ! -d "$MODEL_DIR/transformer" ]; then
  echo "Cosmos3-Nano 权重不完整: $MODEL_DIR" >&2
  exit 1
fi

log "========== python venv (donor 3.10, ABI 对齐 torch) =========="
if [ ! -x "${ENV_DIR}/bin/python" ]; then
  "$DONOR_PY" -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel
log "python=$(python --version) exe=$(which python)"

log "========== pytorch from donor env =========="
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log "torch already ok: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
else
  DONOR_SITE="${DONOR_ENV}/lib/python3.10/site-packages"
  DEST_SITE="${ENV_DIR}/lib/python3.10/site-packages"
  python -m pip install typing-extensions filelock fsspec jinja2 networkx sympy
  for pkg in torch torchvision torchaudio nvidia triton torchgen functorch; do
    if [ -e "${DONOR_SITE}/${pkg}" ]; then
      log "copy ${pkg}"
      rm -rf "${DEST_SITE}/${pkg}"
      cp -a "${DONOR_SITE}/${pkg}" "${DEST_SITE}/${pkg}"
    fi
    for meta in "${DONOR_SITE}"/${pkg}*.dist-info "${DONOR_SITE}"/${pkg}*.egg-info; do
      [ -e "$meta" ] || continue
      rm -rf "${DEST_SITE}/$(basename "$meta")"
      cp -a "$meta" "${DEST_SITE}/$(basename "$meta")"
    done
  done
  for dir in "${DONOR_SITE}"/nvidia_*; do
    [ -e "$dir" ] || continue
    name=$(basename "$dir")
    rm -rf "${DEST_SITE}/${name}"
    cp -a "$dir" "${DEST_SITE}/${name}"
  done
fi
python - <<'PY' | tee -a "$LOG_FILE"
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"gpu0={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
PY

log "========== cosmos python deps =========="
pip install \
  "huggingface_hub" \
  accelerate \
  av \
  einops \
  imageio \
  imageio-ffmpeg \
  pillow \
  "safetensors>=0.4.0" \
  sentencepiece \
  protobuf \
  omegaconf \
  pydantic \
  tyro \
  loguru \
  websockets \
  msgpack \
  "transformers>=4.57.1"

if [ -d "${DIFFUSERS_DIR}/src/diffusers" ]; then
  log "installing diffusers from ${DIFFUSERS_DIR}"
  pip install -e "${DIFFUSERS_DIR}"
else
  pip install "diffusers>=0.35.0"
fi

pip install cosmos_guardrail || log "cosmos_guardrail skipped (T2I/policy 可用 --no-safety-check)"

if [ -d "${FRAMEWORK_DIR}/cosmos_framework" ]; then
  log "========== cosmos-framework editable =========="
  pip install -e "${FRAMEWORK_DIR}" --no-deps || log "framework editable install skipped"
fi

log "========== import check =========="
python - <<'PY' | tee -a "$LOG_FILE"
import importlib
for m in ["torch", "torchvision", "transformers", "diffusers", "huggingface_hub", "accelerate", "safetensors"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"FAIL {m}: {e}")
try:
    from diffusers import Cosmos3OmniPipeline
    print("OK Cosmos3OmniPipeline")
except Exception as e:
    print(f"WARN Cosmos3OmniPipeline: {e}")
try:
    import cosmos_framework
    print("OK cosmos_framework", getattr(cosmos_framework, "__file__", ""))
except Exception as e:
    print(f"WARN cosmos_framework: {e}")
PY

cat > "${ROOT}/env_infer.sh" <<EOF
# source /home/xk/VLA/cosmos3_staging/env_infer.sh
export PATH="\$HOME/.local/bin:\$PATH"
export HF_ENDPOINT="${HF_ENDPOINT}"
export HF_HUB_DISABLE_XET=1
export COSMOS_DIR="${COSMOS_DIR}"
export COSMOS3_MODEL="${MODEL_DIR}"
export COSMOS3_DIFFUSERS="${DIFFUSERS_DIR}"
export COSMOS3_FRAMEWORK="${FRAMEWORK_DIR}"
source "${ENV_DIR}/bin/activate"
echo "python=\$(which python) \$(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
EOF
log "wrote ${ROOT}/env_infer.sh"
log "========== setup done =========="
du -sh "$ENV_DIR" "$MODEL_DIR" 2>/dev/null | tee -a "$LOG_FILE" || true
