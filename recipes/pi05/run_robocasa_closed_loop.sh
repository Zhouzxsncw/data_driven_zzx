#!/usr/bin/env bash
# 扫盘上的 RoboCasa365 checkpoint，把还没评完的接上。
# 一份权重评完是 18 个任务各 30 条，而且 eval_summary.json 里有 success_rate。
# 名单不写死。评的过程中新落到盘上的完整权重，这一轮结束前会再扫一次。
set -u
STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$STAGING/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$STAGING/_pi05_common.sh"

export PI05_EMBODIMENT=robocasa365
export ROBOCASA_TRIALS="${ROBOCASA_TRIALS:-30}"
export ROBOCASA_SPLIT="${ROBOCASA_SPLIT:-target}"
export ROBOCASA_MAX_STEPS="${ROBOCASA_MAX_STEPS:-400}"
export ROBOCASA_REPLAN="${ROBOCASA_REPLAN:-5}"
export ROBOCASA_OBJ_REGISTRIES="${ROBOCASA_OBJ_REGISTRIES:-objaverse,lightwheel}"

RC365_ROOT="${RC365_ROOT:-/home/cigai22/data/rc365}"
EXPECT_BYTES="${RC365_CKPT_BYTES:-7473091464}"
TASK_TIMEOUT="${RC365_TASK_TIMEOUT:-2700}"
LOG=recipes/pi05/server_pi05_robocasa365.log
DRIVER_LOG="${RC365_DRIVER_LOG:-$RC365_ROOT/eval/driver_scan.log}"

TASKS=(
  OpenDrawer
  PickPlaceSinkToCounter
  TurnOnMicrowave
  CloseFridge
  TurnOnSinkFaucet
  NavigateKitchen
  OpenCabinet
  PickPlaceCounterToCabinet
  CoffeeSetupMug
  TurnOnElectricKettle
  OpenStandMixerHead
  CloseToasterOvenDoor
  SlideDishwasherRack
  PickPlaceToasterToCounter
  PickPlaceCounterToStove
  PickPlaceDrawerToCounter
  TurnOffStove
  CloseBlenderLid
)

list_pending() {
  RC365_ROOT="$RC365_ROOT" EXPECT_BYTES="$EXPECT_BYTES" python3 - "${TASKS[@]}" << 'PY'
import json, os, sys
from pathlib import Path

tasks = sys.argv[1:]
root = Path(os.environ["RC365_ROOT"])
expect = int(os.environ["EXPECT_BYTES"])

def k_num(tag):
    digits = "".join(ch for ch in tag if ch.isdigit())
    return int(digits) if digits else 10**9

def task_done(path):
    """有数字结果，或已经记过失败，都算这一档里的这个任务处理过。"""
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if int(data.get("trials") or 0) != int(os.environ.get("ROBOCASA_TRIALS") or 30):
        return False
    if data.get("status") == "failed":
        return True
    return data.get("success_rate") is not None

found = []
for weights in sorted(root.glob("rc365_PI05_*/[0-9]*/pretrained_model/model.safetensors")):
    step_dir = weights.parents[1]
    exp_dir = weights.parents[2]
    name = exp_dir.name
    prefix = "rc365_PI05_"
    tag = name[len(prefix):] if name.startswith(prefix) else name
    step = step_dir.name
    stats = step_dir / "assets" / "norm_stats.json"
    size = weights.stat().st_size
    if size != expect or not stats.is_file():
        print(f"SKIP_PARTIAL {tag} {step} bytes={size} stats={int(stats.is_file())}", file=sys.stderr)
        continue
    found.append((k_num(tag), int(step), tag, step, weights.parent))

by_tag = {}
for item in found:
    by_tag.setdefault(item[2], []).append(item[3])
for tag in by_tag:
    by_tag[tag].sort()

for _, _, tag, step, ckpt in sorted(found):
    legacy = root / "eval" / tag
    recorded = ""
    ckpt_txt = legacy / "ckpt.txt"
    if ckpt_txt.is_file():
        recorded = ckpt_txt.read_text().strip()
    if recorded == str(ckpt) or (not recorded and by_tag[tag] == [step]):
        out = legacy
    elif not recorded and by_tag[tag][0] == step:
        out = legacy
    else:
        out = root / "eval" / f"{tag}_{step}"
    missing = [task for task in tasks if not task_done(out / task / "eval_summary.json")]
    if missing:
        print(f"{tag}\t{step}\t{ckpt}\t{out}\t{len(missing)}")
PY
}

ensure_server() {
  local ckpt=$1
  if pgrep -f "$STAGING/serve_tong_pi05.py" >/dev/null 2>&1 \
     && [[ -f "$LOG" ]] && grep -q "$ckpt" "$LOG"; then
    echo "SERVER_KEEP $ckpt"
    return 0
  fi
  echo "SERVER_START $ckpt $(date -Is)"
  PI05_ROBOCASA_CKPT=$ckpt bash "$STAGING/launch_pi05_server.sh" || true
  local i
  for i in $(seq 1 90); do
    if [[ -f "$LOG" ]] && grep -q "server listening" "$LOG" && grep -q "$ckpt" "$LOG"; then
      echo "SERVER_UP $ckpt"
      return 0
    fi
    if ! pgrep -f "$STAGING/serve_tong_pi05.py" >/dev/null 2>&1; then
      echo "SERVER_DEAD"
      tail -n 30 "$LOG" || true
      return 1
    fi
    sleep 5
  done
  echo "SERVER_TIMEOUT"
  tail -n 30 "$LOG" || true
  return 1
}

record_fail() {
  local out=$1 task=$2 rc=$3
  mkdir -p "$out/$task"
  if [[ -f "$out/$task/eval_summary.json" ]]; then
    return 0
  fi
  python3 - "$out/$task/eval_summary.json" "$task" "$rc" << 'PY'
import json, sys
path, task, rc = sys.argv[1:]
json.dump({
    "sim": "robocasa365",
    "env_name": task,
    "robot": "PandaOmron",
    "trials": 30,
    "success_rate": None,
    "status": "failed",
    "returncode": int(rc),
}, open(path, "w"), ensure_ascii=False, indent=2)
print(f"RECORDED_FAIL {task} rc={rc}")
PY
}

run_one() {
  local tag=$1 step=$2 ckpt=$3 out=$4
  mkdir -p "$out"
  printf "%s\n" "$ckpt" > "$out/ckpt.txt"
  if ! ensure_server "$ckpt"; then
    echo "SERVER_FAIL $tag $step $(date -Is)"
    return 1
  fi
  local task rc
  for task in "${TASKS[@]}"; do
    if [[ -f "$out/$task/eval_summary.json" ]] && python3 - "$out/$task/eval_summary.json" "$tag" "$task" << 'PY'
import json, sys
path, tag, task = sys.argv[1:]
data = json.load(open(path))
trials = int(data.get("trials") or 0)
done = trials == 30 and (data.get("status") == "failed" or data.get("success_rate") is not None)
if not done:
    raise SystemExit(1)
print(f"SKIP {tag} {task}")
PY
    then
      continue
    fi
    echo "START $tag $step $task $(date -Is)"
    mkdir -p "$out/$task"
    set +e
    timeout --signal=TERM "$TASK_TIMEOUT" env \
      ROBOCASA_ENV="$task" ROBOCASA_SAVE_DIR="$out/$task" \
      bash "$STAGING/launch_robocasa_client.sh" > "$out/$task.log" 2>&1
    rc=$?
    set -e
    if [[ "$rc" -eq 0 && -f "$out/$task/eval_summary.json" ]]; then
      python3 - "$out/$task/eval_summary.json" << 'PY'
import json, sys
s = json.load(open(sys.argv[1]))
rate = s.get("success_rate")
trials = int(s.get("trials") or 0)
if rate is None:
    print(f"RATE {s.get('env_name')} failed")
else:
    print(f"RATE {s.get('env_name')} {float(rate):.3f} ({int(round(float(rate)*trials))}/{trials})")
PY
    else
      echo "FAIL $tag $step $task rc=$rc $(date -Is)"
      record_fail "$out" "$task" "$rc"
    fi
    echo "DONE $tag $step $task $(date -Is)"
  done
}

if [[ "${RC365_EVAL_DRY:-}" == "1" ]]; then
  list_pending
  exit 0
fi

if pgrep -f "eval_robocasa_client.py" >/dev/null 2>&1 \
   || pgrep -f "data/rc365/eval/run_closed_loop" >/dev/null 2>&1; then
  echo "ALREADY_RUNNING $(date -Is)"
  exit 0
fi

mkdir -p "$(dirname "$DRIVER_LOG")"
pass=0
while [[ "$pass" -lt 3 ]]; do
  pass=$((pass + 1))
  mapfile -t PENDING < <(list_pending)
  if [[ "${#PENDING[@]}" -eq 0 ]]; then
    echo "QUEUE_EMPTY pass=$pass $(date -Is)"
    break
  fi
  echo "QUEUE pass=$pass count=${#PENDING[@]} $(date -Is)"
  for row in "${PENDING[@]}"; do
    IFS=$'\t' read -r tag step ckpt out missing <<< "$row"
    echo "MODEL $tag step=$step missing=$missing $ckpt"
    run_one "$tag" "$step" "$ckpt" "$out"
  done
done
echo "SCAN_DONE $(date -Is)"
