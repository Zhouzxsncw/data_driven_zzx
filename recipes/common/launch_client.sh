#!/usr/bin/env bash
# 统一真机客户端：只换 YAML（或下面的短名）。
#
#   bash recipes/common/launch_client.sh gwp
#   bash recipes/common/launch_client.sh infer/infer/config/lingbot/gwp_tongbot_16d_00073_local.yaml
#   bash recipes/common/launch_client.sh --list
#   bash recipes/common/launch_client.sh --resolve gwp
#   bash recipes/common/launch_client.sh --open gwp
#
# 相机 / 双臂 / 夹爪 / device health / 脚踏都走 infer/cli/run_batch_infer.py。
# 服务端仍按模型各自 launch_*_server.sh。
# 启动前会把客户端 YAML 弹到 Cursor 编辑器。
#
# 环境变量:
#   INFER_ROOT                 默认本仓 infer/，其次 /home/xk/VLA/infer
#   OPENPI_INFER_PYTHON        默认 conda lingbotva
#   SKIP_PORT_CHECK=1          不探本地服务端口
#   SKIP_YAML_EDIT=1           不弹 YAML
#   DISPLAY                    默认 :1
set -euo pipefail
unalias -a 2>/dev/null || true

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$COMMON_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$COMMON_DIR/open_infer_yaml.sh"

_usage() {
  cat >&2 <<EOF
用法: bash recipes/common/launch_client.sh <yaml|短名>
      bash recipes/common/launch_client.sh --list
      bash recipes/common/launch_client.sh --resolve <yaml|短名>
      bash recipes/common/launch_client.sh --open <yaml|短名>

  bash recipes/common/launch_client.sh gwp
  bash recipes/common/launch_client.sh --list

启动前会把客户端 YAML 弹到 Cursor 编辑器。
旧的 recipes/<model>/launch_*_client.sh 仍可用，内部转到本脚本。
EOF
  exit 2
}

# 短名 → 相对仓库根的 YAML。换模型只改这一处映射 + YAML 本身。
_alias_yaml() {
  case "$1" in
    acot|acot_vla) echo "infer/infer/config/lingbot/acot_vla_tongbot_16d_00036_local.yaml" ;;
    va|va00073|lingbot_va|lingbot_va_00073) echo "infer/infer/config/lingbot/lingbot_va_tongbot_16d_00073_local.yaml" ;;
    va00036|lingbot_va_00036) echo "infer/infer/config/lingbot/lingbot_va_tongbot_16d_00036_local.yaml" ;;
    dreamzero|dz) echo "infer/infer/config/lingbot/dreamzero_tongbot_16d_00073_local.yaml" ;;
    cosmos3|cosmos3_00073) echo "infer/infer/config/lingbot/cosmos3_tongbot_16d_00073_local.yaml" ;;
    cosmos3_00036) echo "infer/infer/config/lingbot/cosmos3_tongbot_16d_00036_local.yaml" ;;
    cosmos3_00091) echo "infer/infer/config/lingbot/cosmos3_tongbot_16d_00091_local.yaml" ;;
    cosmos3_grid) echo "infer/infer/config/lingbot/cosmos3_tongbot_16d_00073_grid.yaml" ;;
    internvla) echo "infer/infer/config/lingbot/internvla_tongbot_16d_00036_local.yaml" ;;
    memoryvla) echo "infer/infer/config/lingbot/memoryvla_tongbot_16d_00086_local.yaml" ;;
    walloss|walloss_00091) echo "infer/infer/config/lingbot/walloss_tongbot_ee_00091_local.yaml" ;;
    walloss_00086) echo "infer/infer/config/lingbot/walloss_tongbot_ee_00086_local.yaml" ;;
    walloss_00036) echo "infer/infer/config/lingbot/walloss_tongbot_ee_00036_s40000_local.yaml" ;;
    walloss_00036_js16) echo "infer/infer/config/lingbot/walloss_tongbot_ee_00036_js16_ep9_local.yaml" ;;
    walloss_00073|walloss_ah8) echo "infer/infer/config/lingbot/walloss_tongbot_ee_00073_ah8_s10000_local.yaml" ;;
    pi05_00086_js16_22445) echo "infer/infer/config/lerobot/pi05_00086_4gpu_bs384_ah100_step22445_ws.yaml" ;;
    pi05_00036_js8_ah15) echo "infer/infer/config/lerobot/pi05_00036_js8_bs384_ah15_fixbase_step7860_ws.yaml" ;;
    pi05_00036_js8_ah100) echo "infer/infer/config/lerobot/pi05_00036_js8_bs384_ah100_fixbase_step7860_ws.yaml" ;;
    pi05_00108_rgb_depth_head|pi05_00108_head) echo "infer/infer/config/lerobot/pi05_00108_a2npu8_ah50_rgb_depth_head_fixbase_step4000_ws.yaml" ;;
    pi05_00108_rgb_depth_head_2000|pi05_00108_head_2000) echo "infer/infer/config/lerobot/pi05_00108_a2npu8_ah50_rgb_depth_head_fixbase_step2000_ws.yaml" ;;
    pi05_00108_rgb_depth_head_ft|pi05_00108_head_ft) echo "infer/infer/config/lerobot/pi05_00108_js16c_ah50_rgb_depth_head_ft_fixbase_step630_ws.yaml" ;;
    pi05_00108_rgb|pi05_00108_rgb_2000) echo "infer/infer/config/lerobot/pi05_00108_npu8_ah50_rgb_fixbase_step2000_ws.yaml" ;;
    pi05_00108_rgb_ft|pi05_00108_rgb_ft_2000) echo "infer/infer/config/lerobot/pi05_00108_npu8_ah50_rgb_ft_fixbase_step2000_ws.yaml" ;;
    pi05_00108_rgb_depth_left|pi05_00108_left) echo "infer/infer/config/lerobot/pi05_00108_js16b_ah50_rgb_depth_left_fixbase_step630_ws.yaml" ;;
    pi05_00108_rgb_depth_left_ft|pi05_00108_left_ft) echo "infer/infer/config/lerobot/pi05_00108_js16c_ah50_rgb_depth_left_ft_fixbase_step630_ws.yaml" ;;
    pi05_00108_rgb_depth_right|pi05_00108_right) echo "infer/infer/config/lerobot/pi05_00108_js16b_ah50_rgb_depth_right_fixbase_step630_ws.yaml" ;;
    fastwam|fastwam_00073) echo "infer/infer/config/lingbot/fastwam_tongbot_16d_00073_local.yaml" ;;
    fastwam_00036) echo "infer/infer/config/lingbot/fastwam_tongbot_16d_00036_local.yaml" ;;
    motus) echo "infer/infer/config/lingbot/motus_tongbot_16d_00073_local.yaml" ;;
    gwp|gigaworld|gwp05) echo "infer/infer/config/lingbot/gwp_tongbot_16d_00073_local.yaml" ;;
    tau|tau0|tau0wm) echo "infer/infer/config/lingbot/tau0wm_tongbot_ee_00036_local.yaml" ;;
    pi05) echo "infer/infer/config/lerobot/pi05_openpi_tongbot_16d_00092_ws_client.yaml" ;;
    pi05_00086_js16_22445) echo "infer/infer/config/lerobot/pi05_00086_4gpu_bs384_ah100_step22445_ws.yaml" ;;
    pi05_00036_js8_ah15) echo "infer/infer/config/lerobot/pi05_00036_js8_bs384_ah15_fixbase_step7860_ws.yaml" ;;
    pi05_00036_js8_ah100) echo "infer/infer/config/lerobot/pi05_00036_js8_bs384_ah100_fixbase_step7860_ws.yaml" ;;
    pi05_00108_rgb_depth_head|pi05_00108_head) echo "infer/infer/config/lerobot/pi05_00108_a2npu8_ah50_rgb_depth_head_fixbase_step4000_ws.yaml" ;;
    pi05_00108_rgb_depth_head_2000|pi05_00108_head_2000) echo "infer/infer/config/lerobot/pi05_00108_a2npu8_ah50_rgb_depth_head_fixbase_step2000_ws.yaml" ;;
    pi05_00108_rgb_depth_head_ft|pi05_00108_head_ft) echo "infer/infer/config/lerobot/pi05_00108_js16c_ah50_rgb_depth_head_ft_fixbase_step630_ws.yaml" ;;
    pi05_00108_rgb|pi05_00108_rgb_2000) echo "infer/infer/config/lerobot/pi05_00108_npu8_ah50_rgb_fixbase_step2000_ws.yaml" ;;
    pi05_00108_rgb_ft|pi05_00108_rgb_ft_2000) echo "infer/infer/config/lerobot/pi05_00108_npu8_ah50_rgb_ft_fixbase_step2000_ws.yaml" ;;
    pi05_00108_rgb_depth_left|pi05_00108_left) echo "infer/infer/config/lerobot/pi05_00108_js16b_ah50_rgb_depth_left_fixbase_step630_ws.yaml" ;;
    pi05_00108_rgb_depth_left_ft|pi05_00108_left_ft) echo "infer/infer/config/lerobot/pi05_00108_js16c_ah50_rgb_depth_left_ft_fixbase_step630_ws.yaml" ;;
    pi05_00108_rgb_depth_right|pi05_00108_right) echo "infer/infer/config/lerobot/pi05_00108_js16b_ah50_rgb_depth_right_fixbase_step630_ws.yaml" ;;
    pi0fast) echo "infer/infer/config/lerobot/pi0fast_openpi_tongbot_16d_00036_ws_client.yaml" ;;
    vla|lingbot_vla) echo "infer/infer/config/lingbot/lingbot_vla_tongbot_16d_00086_local.yaml" ;;
    gr00t|gr00t_00073) echo "infer/infer/config/lingbot/gr00t_tongbot_16d_00073.yaml" ;;
    gr00t_00036) echo "infer/infer/config/lingbot/gr00t_tongbot_16d_00036.yaml" ;;
    *) echo "" ;;
  esac
}

_list_aliases() {
  cat <<EOF
短名                 YAML
acot                 acot_vla_tongbot_16d_00036_local.yaml
va / va00073         lingbot_va_tongbot_16d_00073_local.yaml
va00036              lingbot_va_tongbot_16d_00036_local.yaml
dreamzero            dreamzero_tongbot_16d_00073_local.yaml
cosmos3              cosmos3_tongbot_16d_00073_local.yaml
cosmos3_00036        cosmos3_tongbot_16d_00036_local.yaml
cosmos3_00091        cosmos3_tongbot_16d_00091_local.yaml
cosmos3_grid         cosmos3_tongbot_16d_00073_grid.yaml
internvla            internvla_tongbot_16d_00036_local.yaml
memoryvla            memoryvla_tongbot_16d_00086_local.yaml
walloss              walloss_tongbot_ee_00091_local.yaml
walloss_00086        walloss_tongbot_ee_00086_local.yaml
walloss_00036        walloss_tongbot_ee_00036_s40000_local.yaml
walloss_00036_js16   walloss_tongbot_ee_00036_js16_ep9_local.yaml
walloss_00073        walloss_tongbot_ee_00073_ah8_s10000_local.yaml
pi05_00086_js16_22445  pi05_00086_4gpu_bs384_ah100_step22445_ws.yaml
pi05_00036_js8_ah15  pi05_00036_js8_bs384_ah15_fixbase_step7860_ws.yaml
pi05_00036_js8_ah100 pi05_00036_js8_bs384_ah100_fixbase_step7860_ws.yaml
pi05_00108_rgb_depth_head  pi05_00108_a2npu8_ah50_rgb_depth_head_fixbase_step4000_ws.yaml
pi05_00108_rgb_depth_head_2000  pi05_00108_a2npu8_ah50_rgb_depth_head_fixbase_step2000_ws.yaml
pi05_00108_rgb_depth_head_ft  pi05_00108_js16c_ah50_rgb_depth_head_ft_fixbase_step630_ws.yaml
pi05_00108_rgb  pi05_00108_npu8_ah50_rgb_fixbase_step2000_ws.yaml
pi05_00108_rgb_ft  pi05_00108_npu8_ah50_rgb_ft_fixbase_step2000_ws.yaml
pi05_00108_rgb_depth_left  pi05_00108_js16b_ah50_rgb_depth_left_fixbase_step630_ws.yaml
pi05_00108_rgb_depth_left_ft  pi05_00108_js16c_ah50_rgb_depth_left_ft_fixbase_step630_ws.yaml
pi05_00108_rgb_depth_right  pi05_00108_js16b_ah50_rgb_depth_right_fixbase_step630_ws.yaml
fastwam              fastwam_tongbot_16d_00073_local.yaml
motus                motus_tongbot_16d_00073_local.yaml
gwp                  gwp_tongbot_16d_00073_local.yaml
tau                  tau0wm_tongbot_ee_00036_local.yaml
pi05                 pi05_openpi_tongbot_16d_00092_ws_client.yaml
pi0fast              pi0fast_openpi_tongbot_16d_00036_ws_client.yaml
vla                  lingbot_vla_tongbot_16d_00086_local.yaml
gr00t                gr00t_tongbot_16d_00073.yaml
EOF
}

_resolve_config() {
  local p="$1"
  local alias_rel
  alias_rel="$(_alias_yaml "$p")"
  if [[ -n "$alias_rel" ]]; then
    p="$alias_rel"
  fi
  local candidates=(
    "$p"
    "$PWD/$p"
    "$REPO_ROOT/$p"
    "$REPO_ROOT/infer/$p"
    "$REPO_ROOT/infer/infer/config/$p"
    "$REPO_ROOT/infer/infer/config/lingbot/$p"
    "$REPO_ROOT/infer/infer/config/lerobot/$p"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$c"
      return 0
    fi
  done
  return 1
}

_die_missing_yaml() {
  echo "[ERROR] 找不到客户端 YAML: ${1:-<empty>}" >&2
  echo "        bash recipes/common/launch_client.sh --list" >&2
  exit 1
}

if [[ $# -lt 1 ]]; then
  _usage
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  _usage
fi
if [[ "$1" == "--list" || "$1" == "-l" ]]; then
  _list_aliases
  exit 0
fi
if [[ "$1" == "--resolve" ]]; then
  shift || true
  CONFIG="$(_resolve_config "${1:-}" || true)"
  [[ -n "${CONFIG:-}" ]] || _die_missing_yaml "${1:-}"
  printf '%s\n' "$CONFIG"
  exit 0
fi
if [[ "$1" == "--open" ]]; then
  shift || true
  CONFIG="$(_resolve_config "${1:-}" || true)"
  [[ -n "${CONFIG:-}" ]] || _die_missing_yaml "${1:-}"
  open_infer_yaml "$CONFIG"
  exit 0
fi

CONFIG_ARG="$1"
shift || true

CONFIG="$(_resolve_config "$CONFIG_ARG" || true)"
if [[ -z "${CONFIG:-}" ]]; then
  _die_missing_yaml "$CONFIG_ARG"
fi
open_infer_yaml "$CONFIG"

if [[ -z "${INFER_ROOT:-}" ]]; then
  if [[ -f "$REPO_ROOT/infer/infer/cli/run_batch_infer.py" ]]; then
    INFER_ROOT="$REPO_ROOT/infer"
  elif [[ -f /home/xk/VLA/infer/infer/cli/run_batch_infer.py ]]; then
    INFER_ROOT="/home/xk/VLA/infer"
  else
    echo "[ERROR] 找不到 infer/cli/run_batch_infer.py。设 INFER_ROOT。" >&2
    exit 1
  fi
fi
INFER_ROOT="$(python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$INFER_ROOT")"

PYTHON="${OPENPI_INFER_PYTHON:-${LINGBOTVA_PYTHON:-/home/xk/miniconda3/envs/lingbotva/bin/python}}"
if [[ ! -x "$PYTHON" ]]; then
  echo "[ERROR] 找不到客户端 Python: $PYTHON" >&2
  exit 1
fi

CQ_SRC="$REPO_ROOT/legacy/CQTongBot_infer/src"
if [[ ! -d "$CQ_SRC" ]]; then
  CQ_SRC="/home/xk/CQTongBot_infer/src"
fi

HOST_PORT="$("$PYTHON" - "$CONFIG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")

def first(key: str) -> str:
    m = re.search(rf"(?m)^[ \t]*{re.escape(key)}:[ \t]*([^\n#]+)", text)
    if not m:
        return ""
    return m.group(1).strip().strip("\"'")

host = first("lingbot_host") or first("gr00t_host") or "127.0.0.1"
port = first("lingbot_port") or first("gr00t_port") or ""
ptype = first("type")
print(host)
print(port)
print(ptype)
PY
)"
HOST="$(printf '%s\n' "$HOST_PORT" | sed -n '1p')"
PORT="$(printf '%s\n' "$HOST_PORT" | sed -n '2p')"
PTYPE="$(printf '%s\n' "$HOST_PORT" | sed -n '3p')"

_port_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -H -ltn "sport = :${port}" 2>/dev/null | grep -q .
    return
  fi
  netstat -ltn 2>/dev/null | grep -E ":${port}[[:space:]]" | grep -q .
}

_is_loopback() {
  case "$1" in
    127.0.0.1|localhost|::1) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "${SKIP_PORT_CHECK:-0}" != "1" && -n "$PORT" ]] && _is_loopback "$HOST"; then
  if ! _port_listening "$PORT"; then
    echo "[ERROR] 服务 ${HOST}:${PORT} 还没在听（policy.type=${PTYPE:-?}）。" >&2
    echo "        先起对应 launch_*_server.sh，日志出现 listening 后再开客户端。" >&2
    echo "        跳过探测: SKIP_PORT_CHECK=1 bash recipes/common/launch_client.sh ..." >&2
    exit 1
  fi
fi

export DISPLAY="${DISPLAY:-:1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${CQ_SRC}:${INFER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$INFER_ROOT/logs"
cd "$INFER_ROOT"

echo "=== 统一真机客户端 ==="
echo "Python: $PYTHON ($("$PYTHON" --version 2>&1))"
echo "Config: $CONFIG"
echo "INFER_ROOT: $INFER_ROOT"
echo "DISPLAY: $DISPLAY"
echo "Server: ${HOST}:${PORT:-?}  type=${PTYPE:-?}"
echo "======================"
exec "$PYTHON" infer/cli/run_batch_infer.py --config "$CONFIG" "$@"
