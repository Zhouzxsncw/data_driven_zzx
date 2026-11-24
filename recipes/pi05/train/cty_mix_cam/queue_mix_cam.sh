#!/usr/bin/env bash
# One column of the π0.5 camera ablation. The next yaml starts only after the
# previous torchrun exits 0. An existing output dir is left alone.
set -u

REPO=/workspace/openpi/cigai_train/openpi-jiangsuan
cd "${REPO}"

HOST_TAG="${1:?host tag}"
shift
if [[ "$#" -lt 1 ]]; then
  echo "[queue] no yamls"
  exit 1
fi

LOCK=/workspace/openpi/train_mix_cam_queue.lock
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "[queue] another queue holds ${LOCK}"
  exit 1
fi

if pgrep -af "train_from_yaml.py" | grep -v pgrep >/dev/null; then
  echo "[queue] train_from_yaml already running, not starting another"
  pgrep -af "train_from_yaml.py" | grep -v pgrep || true
  exit 1
fi

DONE="/workspace/openpi/train_mix_cam_queue_${HOST_TAG}.done"
mkdir -p "$(dirname "${DONE}")"

for yaml in "$@"; do
  if [[ -f "${DONE}" ]] && grep -qx "${yaml}" "${DONE}"; then
    echo "[queue] skip done ${yaml}"
    continue
  fi
  exp="$(python3 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1],encoding="utf-8"))["exp_name"])' "${yaml}")"
  out="/workspace/outputs/${exp}"
  if [[ -d "${out}" ]]; then
    echo "[queue] output exists, refuse to restart ${out}"
    exit 1
  fi
  log="/workspace/openpi/train_${exp}.log"
  echo "[queue] start $(date -Is) ${yaml} -> ${log}"
  YAML="${yaml}" GPUS=0,1,2,3,4,5,6,7 PORT=29701 NPROC=8 LOG="${log}" \
    bash scripts/start_train_ppu8.sh
  rc=$?
  echo "[queue] exit ${rc} $(date -Is) ${yaml}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "[queue] stop on failure"
    exit "${rc}"
  fi
  echo "${yaml}" >> "${DONE}"
done

echo "[queue] column finished $(date -Is)"
