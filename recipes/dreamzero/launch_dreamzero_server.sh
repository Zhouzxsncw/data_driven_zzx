#!/usr/bin/env bash
# DreamZero 00073 真机服务端（TongBot 16D，协议对齐 LingBot-VA 客户端）
# 默认权重: H100 SFT DreamZero-14B global_step_20000（00073 收官，约 41G 显存 + KV cache）
#           回退早期: DREAMZERO_CKPT=.../dreamzero_14b/global_step_10000/.../infer_weights.pt
# 切回 5B:  DREAMZERO_CKPT=.../dreamzero_5b/global_step_3000/.../infer_weights.pt \
#           DREAMZERO_CONFIG=tong00073_infer_dreamzero_5b_local bash $0
# 端口 1235，避免和正在跑的 VA 00036（1234）冲突。
#
# GPU 只有一张（98G）：14B 全量 bf16 ≈ 41G 起，VA 00036 若占 23GB 可并存，
# 但建议真机部署时先停 VA:
#   pkill -f 'run_lingbotva_infer.py --config .*config_00036'
#
#   bash recipes/dreamzero/launch_dreamzero_server.sh
#
#   # 开启 dream video latent 落盘（每包捕获 video_pred .pt，离线 VAE decode）：
#   DREAMZERO_DUMP_VIDEO=/home/xk/VLA/h100_ckpts/dreamzero_dream bash $0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 清掉 env_infer 之外可能残留的用户 site
unset PYTHONNOUSERSITE || true
# shellcheck disable=SC1091
source "$ROOT/env_infer.sh"
export PYTHONPATH="/home/xk/VLA/RLinf:${DREAMZERO_PATH}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT="${DREAMZERO_CKPT:-/home/xk/VLA/h100_ckpts/dreamzero_14b/global_step_20000/actor/model_state_dict/infer_weights.pt}"
CONFIG="${DREAMZERO_CONFIG:-tong00073_infer_dreamzero_14b_local}"
PORT="${DREAMZERO_PORT:-1235}"
LOG="$ROOT/server_dreamzero_00073.log"
PROMPT="${DREAMZERO_PROMPT:-Using both hands, pick up the courier items from the green conveyor belt. Place the white courier bags into the left box and the brown courier boxes into the right box.}"
DUMP_VIDEO="${DREAMZERO_DUMP_VIDEO:-}"  # 非空路径 = 开启 dream video latent 落盘

echo "=== DreamZero 00073 ($CONFIG) ==="
echo "Python: $(which python) ($(python --version 2>&1))"
echo "Ckpt:   $CKPT"
echo "Config: $CONFIG"
echo "Port:   $PORT"
echo "Log:    $LOG"
echo "Dump:   ${DUMP_VIDEO:-off}"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null)"
echo "================================"

if [[ ! -f "$CKPT" ]]; then
  echo "权重不存在: $CKPT" >&2
  exit 1
fi

if ss -tln | grep -q ':1234'; then
  echo "注意: 1234 上已有 VA 服务，占 GPU。DreamZero 建议先停 VA 00036，否则可能 OOM。"
fi

OLD=$(pgrep -f "$ROOT/serve_tong_dreamzero.py" || true)
if [[ -n "${OLD}" ]]; then
  echo "杀掉旧 DreamZero 服务: $OLD"
  echo "$OLD" | xargs kill -9 2>/dev/null || true
  sleep 2
fi

DUMP_ARGS=""
if [[ -n "${DUMP_VIDEO}" ]]; then
  mkdir -p "${DUMP_VIDEO}" 2>/dev/null || true
  DUMP_ARGS="--dump-video ${DUMP_VIDEO}"
fi

# 免改 YAML 临时覆盖 hydra 字段（空格分隔，可多个）:
#   DREAMZERO_SET="data.train_data_paths=/data/x actor.model.metadata_json_path=/data/m.json" bash $0
SET_ARGS=()
for kv in ${DREAMZERO_SET:-}; do SET_ARGS+=(--set "$kv"); done

nohup python -u "$ROOT/serve_tong_dreamzero.py" \
  --ckpt "$CKPT" --config-name "$CONFIG" --port "$PORT" --instruction "$PROMPT" \
  ${DUMP_ARGS} "${SET_ARGS[@]}" \
  > "$LOG" 2>&1 &
PID=$!
echo "Server PID: $PID"
echo "日志: $LOG"
echo "等模型加载（约 1–2 分钟），出现 listening 后再开客户端:"
echo "  bash $ROOT/launch_dreamzero_client.sh"
echo "  tail -f $LOG"
sleep 3
if ps -p "$PID" >/dev/null 2>&1; then
  echo "已启动 PID=$PID"
  bash "$ROOT/../common/start_lower_machine.sh" || \
    echo "[WARN] 下位机未拉起。重试: bash $ROOT/../common/start_lower_machine.sh ；跳过: SKIP_LOWER_MACHINE=1" >&2
else
  echo "启动失败:"
  tail -40 "$LOG"
  exit 1
fi
