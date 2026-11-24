#!/usr/bin/env bash
# PI0.5 环境自检：transformers 分支、tokenizer 缓存、可选 policy 初始化。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." ROOT="$(cd "$(dirname "$0")/.." && pwd)"ROOT="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
# shellcheck source=../train/scripts/lib/sim_offline_env.sh
source "${ROOT}/train/scripts/lib/sim_offline_env.sh"
sim_apply_offline_env

PY="${LEROBOT_PYTHON:-${ROOT}/.venv310/bin/python3.10}"
export PYTHONPATH="${ROOT}/train/src:${PYTHONPATH:-}"
fail() { echo "[verify_pi05] FAIL: $*" >&2; exit 1; }
ok() { echo "[verify_pi05] OK: $*"; }

"$PY" - <<'PY' || fail "transformers 非 PI 分支（需 pip install 'transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi'）"
import transformers
from transformers.models.siglip import check
v = transformers.__version__
assert check.check_whether_transformers_replace_is_installed_correctly(), "siglip replace check failed"
print(f"transformers {v}, siglip replace OK")
PY

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
_paligemma_ok=0
for _d in "${HF_HOME}/hub/models--google--paligemma-3b-pt-224/snapshots"/*/; do
  [[ -f "${_d}tokenizer.json" || -f "${_d}spiece.model" ]] && _paligemma_ok=1 && break
done
[[ "$_paligemma_ok" -eq 1 ]] || fail "paligemma tokenizer 未在 ${HF_HOME}/hub/（见 data/README.md）"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
"$PY" - <<'PY' || fail "paligemma tokenizer 加载失败（检查 data/hf_hub 完整性）"
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
print("paligemma tokenizer cache OK")
PY

if [[ "${VERIFY_PI05_SKIP_POLICY:-false}" != "true" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  "$PY" - <<'PY' || fail "PI05Policy 初始化失败"
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
PI05Policy(PI05Config())
print("PI05Policy init OK")
PY
fi

ok "PI0.5 环境就绪"
