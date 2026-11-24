# 推理启动横幅里打印这一档 checkpoint 的训练 loss。
# 由 launch_client.sh / start_pi05_infer.sh source。不拦截开跑。
# shellcheck shell=bash

_CKPT_LOSS_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_ckpt_loss() {
  local py="$1"
  local cfg="$2"
  local infer_root color=()
  infer_root="$(cd "$_CKPT_LOSS_COMMON/../../infer" && pwd)"
  if [[ -t 1 ]]; then
    color=(--color)
  fi
  PYTHONPATH="${infer_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$py" -m infer.recording.ckpt_loss_notice --config "$cfg" "${color[@]}" \
    || echo "LOSS 未找到（读取失败，不影响推理）"
}
