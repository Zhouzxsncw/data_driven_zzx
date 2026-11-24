#!/usr/bin/env bash
# 4090 本机流水：自己下数据集 → ACT → DP → 闭环评测 → summary.json
# 用法: bash pipeline.sh <sim> <dataset_repo_id>
# 例:   bash pipeline.sh libero HuggingFaceVLA/libero
# 卡上已有进程或剩余磁盘 < 80GB 时直接退出，不杀已有进程。
set -euo pipefail

SIM="${1:?sim name}"
DATASET="${2:?hf dataset repo id}"
ROOT="${HOME}/sim4090/${SIM}"
VENV="${HOME}/sim4090/venv"
STEPS="${SIM4090_STEPS:-20000}"
BATCH="${SIM4090_BATCH:-8}"
N_EP="${SIM4090_EPISODES:-20}"

mkdir -p "${ROOT}"
exec >>"${ROOT}/pipeline.log" 2>&1
echo "=== $(date -Is) sim=${SIM} dataset=${DATASET} ==="

APPS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -viE 'chrome|Xorg|gnome-shell|Xwayland' || true)"
if printf '%s\n' "${APPS}" | grep -q '[0-9]'; then
  echo "SKIP: GPU already has a compute process"
  printf '%s\n' "${APPS}"
  exit 0
fi

AVAIL="$(df -BG --output=avail "${HOME}" | tail -1 | tr -dc '0-9')"
if [ "${AVAIL:-0}" -lt 80 ]; then
  echo "SKIP: free disk ${AVAIL}GB < 80GB"
  exit 0
fi

if [ ! -x "${VENV}/bin/lerobot-train" ]; then
  rm -rf "${VENV}"
  PY="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    if [ ! -x "${HOME}/.local/bin/uv" ]; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="${HOME}/.local/bin:${PATH}"
    uv python install 3.11
    uv venv --python 3.11 --seed "${VENV}"
  elif ! python3 -m venv "${VENV}"; then
    python3 -m venv --without-pip "${VENV}"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "${VENV}/bin/python3" /tmp/get-pip.py
  fi
  "${VENV}/bin/python" -m pip install -U pip
  "${VENV}/bin/python" -m pip install 'lerobot[libero,pusht,robotwin]' || "${VENV}/bin/python" -m pip install 'lerobot'
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export HF_HOME="${HOME}/sim4090/hf"
export WANDB_MODE=disabled
mkdir -p "${HF_HOME}"

echo "--- download ${DATASET} ---"
huggingface-cli download "${DATASET}" --repo-type dataset

train_one() {
  local policy="$1"
  local out="${ROOT}/${policy}"
  if [ -f "${out}/pretrained_model/config.json" ] || [ -d "${out}/checkpoints" ]; then
    echo "SKIP train ${policy}: ${out} already exists"
    return 0
  fi
  echo "--- train ${policy} ---"
  lerobot-train \
    --dataset.repo_id="${DATASET}" \
    --policy.type="${policy}" \
    --policy.device=cuda \
    --output_dir="${out}" \
    --job_name="${SIM}_${policy}" \
    --wandb.enable=false \
    --batch_size="${BATCH}" \
    --steps="${STEPS}" \
    --num_workers=2 \
    --log_freq=100 \
    --save_freq=5000
}

eval_one() {
  local policy="$1"
  local out="${ROOT}/${policy}"
  local ckpt
  ckpt="$(find "${out}" -type d -name pretrained_model | sort | tail -1 || true)"
  if [ -z "${ckpt}" ]; then
    echo "NO CKPT ${policy}"
    return 0
  fi
  local env_type=""
  case "${SIM}" in
    libero) env_type="libero" ;;
    robotwin) env_type="robotwin" ;;
  esac
  if [ -z "${env_type}" ]; then
    echo "EVAL DEFERRED ${policy}: ${SIM} has no lerobot-eval env yet; ckpt kept, success_rate left null"
    python3 - "${ROOT}" "${SIM}" "${policy}" "${DATASET}" "${ckpt}" "${N_EP}" <<'PY'
import json, sys
from pathlib import Path
root, sim, policy, dataset, ckpt, n_ep = sys.argv[1:]
row = {
    "sim": sim, "policy": policy, "dataset": dataset,
    "success_rate": None, "task_completion": None,
    "n_episodes": int(n_ep), "ckpt": ckpt, "eval_exit": None,
    "note": "closed-loop eval not wired for this sim",
}
path = Path(root) / "summary.json"
rows = []
if path.exists():
    try:
        old = json.loads(path.read_text())
        rows = old if isinstance(old, list) else [old]
    except json.JSONDecodeError:
        rows = []
rows = [r for r in rows if r.get("policy") != policy]
rows.append(row)
path.write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(row))
PY
    return 0
  fi
  echo "--- eval ${policy} ckpt=${ckpt} env=${env_type} ---"
  set +e
  lerobot-eval \
    --policy.path="${ckpt}" \
    --env.type="${env_type}" \
    --eval.batch_size=1 \
    --eval.n_episodes="${N_EP}" \
    --output_dir="${ROOT}/eval_${policy}" \
    >"${ROOT}/eval_${policy}.log" 2>&1
  local rc=$?
  set -e
  python3 - "${ROOT}" "${SIM}" "${policy}" "${DATASET}" "${ckpt}" "${rc}" "${N_EP}" <<'PY'
import json, re, sys
from pathlib import Path
root, sim, policy, dataset, ckpt, rc, n_ep = sys.argv[1:]
log = Path(root) / f"eval_{policy}.log"
text = log.read_text(errors="replace") if log.exists() else ""
m = re.search(r"success(?:\s*rate)?[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*%?", text, re.I)
rate = float(m.group(1)) if m else None
if rate is not None and rate > 1:
    rate = rate / 100.0
# 二进制任务：完成度=成功率。CALVIN 若日志里有 avg length，完成度=长度/5。
comp = rate
lm = re.search(r"avg(?:erage)?\s+(?:seq(?:uence)?)?\s*len(?:gth)?[^0-9]*([0-9]+(?:\.[0-9]+)?)", text, re.I)
if lm:
    comp = float(lm.group(1)) / 5.0
row = {
    "sim": sim,
    "policy": policy,
    "dataset": dataset,
    "success_rate": rate,
    "task_completion": comp,
    "n_episodes": int(n_ep),
    "ckpt": ckpt,
    "eval_exit": int(rc),
}
path = Path(root) / "summary.json"
rows = []
if path.exists():
    try:
        old = json.loads(path.read_text())
        rows = old if isinstance(old, list) else [old]
    except json.JSONDecodeError:
        rows = []
rows = [r for r in rows if r.get("policy") != policy]
rows.append(row)
path.write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(row))
PY
}

train_one act
train_one diffusion
eval_one act
eval_one diffusion
echo "=== $(date -Is) done ${SIM} ==="
