#!/bin/bash
# One-process import check. Install pure-python gaps in the venv, never torch/deepspeed.
set -u
source /home/gr00t/code/env_npu.sh >/dev/null
export PYTHONPATH="/home/gr00t/src/Isaac-GR00T:/home/gr00t/shim:/home/gr00t/code:${PYTHONPATH:-}"
cd /home/gr00t/src/Isaac-GR00T
PY=/home/gr00t/venv/bin/python
for i in 1 2 3 4 5 6 7 8; do
  set +e
  out=$($PY -c 'from gr00t.experiment.experiment import run; from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config; print("PREFLIGHT_OK")' 2>&1)
  rc=$?
  set -e
  echo "$out" | grep -E 'PREFLIGHT_OK|No module named|Error' | grep -v ImportWarning | tail -6
  if [ "$rc" -eq 0 ]; then
    echo DONE
    exit 0
  fi
  mod=$(echo "$out" | sed -n "s/.*No module named ['\"]\\([^'\"]*\\)['\"].*/\\1/p" | tail -1)
  echo "ROUND $i MISSING [$mod]"
  if [ -z "$mod" ]; then
    echo "$out" | tail -30
    exit 1
  fi
  case "$mod" in
    tree) pkg=dm-tree ;;
    cv2) pkg=opencv-python-headless ;;
    yaml) pkg=pyyaml ;;
    sklearn) pkg=scikit-learn ;;
    *) pkg=$mod ;;
  esac
  case "$pkg" in
    torch|torch_npu|flash-attn|flash_attn|deepspeed)
      echo "REFUSE $pkg"
      exit 1
      ;;
  esac
  $PY -m pip install --disable-pip-version-check -q "$pkg" || { echo "PIP_FAIL $pkg"; exit 1; }
  echo "INSTALLED $pkg"
done
echo STILL_FAIL
exit 1
