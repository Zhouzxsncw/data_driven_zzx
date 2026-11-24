#!/usr/bin/env bash
# SSH 到下位机，拉起夹爪 + 外骨骼（run.py）。幂等：已在跑则跳过。
#
#   bash recipes/common/start_lower_machine.sh
#
# 各 launch_*_server.sh 在服务进程起来后调用本脚本。
# SKIP_LOWER_MACHINE=1 跳过。EXO_SSH_PASSWORD 默认三个空格（与 configure_servo.py 一致）。
set -euo pipefail

if [[ "${SKIP_LOWER_MACHINE:-0}" == "1" ]]; then
  echo "[lower] SKIP_LOWER_MACHINE=1，跳过下位机"
  exit 0
fi

LOW_HOST="${LOW_HOST:-192.168.1.13}"
LOW_USER="${LOW_USER:-lab}"
REMOTE_DIR="${LOWER_MACHINE_DIR:-/home/lab/TongGenesis/scripts/lower_machine}"
REMOTE_PY="${LOWER_PYTHON:-/home/lab/miniconda3/envs/tong/bin/python}"
REMOTE_LOG="${LOWER_LOG:-/tmp/lower_run.log}"
PASSWORD="${EXO_SSH_PASSWORD:?Set EXO_SSH_PASSWORD}"

detect_high_host() {
  if [[ -n "${HIGH_HOST:-}" ]]; then
    printf '%s\n' "$HIGH_HOST"
    return
  fi
  local ip=""
  ip="$(ip -4 -o addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 \
    | grep -E '^192\.168\.1\.' \
    | grep -vE '^192\.168\.1\.(13|17|18)$' \
    | head -1 || true)"
  printf '%s\n' "${ip:-192.168.1.14}"
}

HIGH_HOST="$(detect_high_host)"

if ! command -v sshpass >/dev/null 2>&1; then
  echo "[lower] [ERROR] 需要 sshpass：sudo apt install sshpass" >&2
  exit 1
fi

echo "[lower] ping ${LOW_HOST} ..."
if ! ping -c 1 -W 2 "${LOW_HOST}" >/dev/null 2>&1; then
  echo "[lower] [ERROR] ${LOW_HOST} 不通。下位机没开机，或本机 USB 网卡没拿到 192.168.1.14。" >&2
  exit 1
fi

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o PasswordAuthentication=yes
  -o PubkeyAuthentication=no
  -o ConnectTimeout=8
  -o LogLevel=ERROR
)

echo "[lower] SSH ${LOW_USER}@${LOW_HOST}  HIGH_HOST=${HIGH_HOST}"

# stdin 交给远端 bash -s；sshpass -e 走 SSHPASS，不占用 stdin。
set +e
out="$(
  SSHPASS="${PASSWORD}" sshpass -e ssh "${SSH_OPTS[@]}" "${LOW_USER}@${LOW_HOST}" "bash -s" <<REMOTE
set -u
DIR=${REMOTE_DIR@Q}
PY=${REMOTE_PY@Q}
HH=${HIGH_HOST@Q}
LOG=${REMOTE_LOG@Q}

if [ ! -x "\$PY" ]; then
  echo "MISSING_PYTHON \$PY"
  exit 2
fi
if [ ! -f "\$DIR/run.py" ]; then
  echo "MISSING_RUN_PY \$DIR/run.py"
  exit 2
fi

gripper=0
exo=0
pgrep -f '[o]nly_gripper.py' >/dev/null 2>&1 && gripper=1
pgrep -f '[e]xo_read_safety.py' >/dev/null 2>&1 && exo=1

if [ "\$gripper" -eq 1 ] && [ "\$exo" -eq 1 ]; then
  echo "ALREADY_RUNNING gripper+exo"
  pgrep -af 'only_gripper.py|exo_read_safety.py|run.py' || true
  exit 0
fi

cd "\$DIR" || exit 2
export HIGH_HOST="\$HH"
export PYTHONUNBUFFERED=1

if [ "\$gripper" -eq 0 ] && [ "\$exo" -eq 0 ]; then
  echo "STARTING run.py HIGH_HOST=\$HH"
  nohup setsid "\$PY" -u run.py >> "\$LOG" 2>&1 < /dev/null &
elif [ "\$gripper" -eq 1 ] && [ "\$exo" -eq 0 ]; then
  echo "GRIPPER_UP starting exo_read_safety.py HIGH_HOST=\$HH"
  nohup setsid "\$PY" -u exo_read_safety.py >> "\$LOG" 2>&1 < /dev/null &
else
  echo "EXO_UP starting only_gripper.py HIGH_HOST=\$HH"
  nohup setsid "\$PY" -u only_gripper.py --skip-arm-init >> "\$LOG" 2>&1 < /dev/null &
fi

for i in 1 2 3 4 5 6 7 8 9 10; do
  g=0
  e=0
  pgrep -f '[o]nly_gripper.py' >/dev/null 2>&1 && g=1
  pgrep -f '[e]xo_read_safety.py' >/dev/null 2>&1 && e=1
  if [ "\$g" -eq 1 ]; then
    echo "UP gripper=\$g exo=\$e HIGH_HOST=\$HH"
    pgrep -af 'only_gripper.py|exo_read_safety.py|run.py' || true
    if [ "\$e" -eq 1 ] || [ "\$i" -ge 6 ]; then
      [ "\$e" -eq 0 ] && echo "WARN exo not up yet, see \$LOG"
      exit 0
    fi
  fi
  sleep 1
done

echo "START_FAILED"
tail -n 50 "\$LOG" || true
exit 1
REMOTE
)"
rc=$?
set -e

printf '%s\n' "$out"
if [[ "$rc" -ne 0 ]]; then
  echo "[lower] [ERROR] SSH/启动失败 rc=${rc}" >&2
  exit 1
fi

if printf '%s\n' "$out" | grep -q '^ALREADY_RUNNING'; then
  echo "[lower] 下位机已在跑，跳过"
elif printf '%s\n' "$out" | grep -q '^UP '; then
  echo "[lower] 下位机已拉起（夹爪监听 ${LOW_HOST}:30015，状态回 ${HIGH_HOST}:30016）"
else
  echo "[lower] [ERROR] 远端输出异常" >&2
  exit 1
fi
