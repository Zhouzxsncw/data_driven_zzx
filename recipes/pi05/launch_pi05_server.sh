#!/usr/bin/env bash
# π₀.₅ websocket 服务端。配置全部来自统一 YAML（PI05_CONFIG），权重来自 PI05_CKPT。
#
#   PI05_CONFIG=/abs/x.yaml PI05_CKPT=/abs/ckpt/pretrained_model \
#     bash recipes/pi05/launch_pi05_server.sh                  # 启动独立副本服务
#   bash recipes/pi05/launch_pi05_server.sh --load-only        # 只加载权重，不占端口、不杀线上进程
#
# 端口取自 YAML 的 policy.lingbot_port；PI05_SERVER_LOG 非空时把日志追加到该文件。
set -euo pipefail
STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$STAGING/../.." && pwd)"
# shellcheck disable=SC1091
source "$STAGING/_pi05_common.sh"

INFER_ROOT="${INFER_ROOT:-${REPO_ROOT}/infer}"
PYTHON="${PI05_PYTHON:-python}"
CONFIG="${PI05_CONFIG:?需要 PI05_CONFIG 指向统一 YAML}"
CKPT="${PI05_CKPT:?需要 PI05_CKPT 指向 pretrained_model}"

[[ -f "$CONFIG" ]] || { echo "[ERROR] 找不到 YAML: $CONFIG" >&2; exit 1; }
[[ -f "$CKPT/model.safetensors" || -d "$CKPT/params" ]] || { echo "[ERROR] 权重未就绪: $CKPT" >&2; exit 1; }

export PYTHONPATH="${INFER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

RUN_ARGS=(--config "$CONFIG" --ckpt "$CKPT")

# --load-only：加载权重后立即退出，不 bind 端口
if [[ "${1:-}" == "--load-only" ]]; then
  exec "$PYTHON" -u "$STAGING/serve_tong_pi05.py" "${RUN_ARGS[@]}" --load-only
fi

# Existing services keep their processes and ports.

echo "[launch] config=$CONFIG"
echo "[launch] ckpt=$CKPT"
if [[ -n "${PI05_SERVER_LOG:-}" ]]; then
  echo "[launch] log -> $PI05_SERVER_LOG"
  exec "$PYTHON" -u "$STAGING/serve_tong_pi05.py" "${RUN_ARGS[@]}" >>"$PI05_SERVER_LOG" 2>&1
fi
exec "$PYTHON" -u "$STAGING/serve_tong_pi05.py" "${RUN_ARGS[@]}"
