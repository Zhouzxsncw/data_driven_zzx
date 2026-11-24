#!/usr/bin/env bash
# pi05 真机推理 · 常驻入口（唯一 recipe，换模型只改 yaml 不新建文件）
#   bash recipes/pi05/run_current_tk.sh check    # 只验导入，不载权重
#   bash recipes/pi05/run_current_tk.sh server   # 起 1297 服务
#   bash recipes/pi05/run_current_tk.sh client   # 起客户端
#   bash recipes/pi05/run_current_tk.sh client --pico-recovery  # 自动准备 PICO 并启用接管
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
YAML="$ROOT/infer/infer/config/lerobot/pi05_current_tk_ws.yaml"
export PI05_SERVER_LOG="$ROOT/recipes/pi05/server_current_tk.log"
export SKIP_LOWER_MACHINE="${SKIP_LOWER_MACHINE:-1}"

# ckpt 直接从 yaml 读，避免和 yaml 脱节
CKPT=$(/home/cigai22/anaconda3/envs/openpi-pi05-infer/bin/python - "$YAML" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['policy']['ckpt_dir'])
PY
)
echo "[run_current_tk] yaml = $YAML"
echo "[run_current_tk] ckpt = $CKPT"

exec bash "$ROOT/recipes/pi05/run_tk_pi05.sh" "${1:-check}" "$YAML" "$CKPT" "${@:2}"
