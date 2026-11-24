#!/usr/bin/env bash
# GigaWorld-Policy「还在跑 / 还会回来」的判定，τ0-WM 的门控靠它。
#
# 为什么不能只看 `pgrep -f scripts/train.py`：
#   GWP 侧有两个看门狗（resume_bump_bs_watch.sh 会在 bs4→bs8 切换时
#   kill 旧进程再拉起新进程；watch_then_launch_train.sh 会在冒烟 READY 后启动六卡）。
#   在 kill→重启 那几分钟窗口里，GWP 训练进程不在、显存也是空的，
#   只看训练进程会误判成「GWP 结束了」，然后 τ0-WM 冲进去把卡抢了 —— 正是用户明令禁止的。
#
#   bash -c 'source h100/gwp_gate.sh; gwp_family_running && echo BUSY || echo FREE'

# 训练本体（含 0_5 / 5b 等所有 config 变体，前缀匹配）
GWP_TRAIN_MATCH="${GWP_TRAIN_MATCH:-scripts/train.py --config giga_world_policy}"
# DeepSpeed/accelerate 另一条入口
GWP_TASK_MATCH="${GWP_TASK_MATCH:-run_task.py --config configs.giga_world_policy}"
# 会重新拉起 GWP 的看门狗
GWP_WATCH_MATCH="${GWP_WATCH_MATCH:-resume_bump_bs_watch.sh|watch_then_launch_train.sh|wait_upload_then_prep.sh}"

# 注意：每条 pgrep 都要 `|| true`。没命中时 pgrep 返回 1，调用方（launch_train.sh）
# 开着 set -euo pipefail，会在这里直接以 rc=1 退出，把后面的 REFUSED 提示和 exit 75 吞掉。
gwp_family_pids() {   # 打印所有命中的 pid（每行一个）
  # pgrep 的模式本身就是 ERE（`|` 直接可用，别加 -E：procps 的 pgrep 没这个选项）
  pgrep -f -a "${GWP_TRAIN_MATCH}" 2>/dev/null | cut -d' ' -f1 || true
  pgrep -f -a "${GWP_TASK_MATCH}"   2>/dev/null | cut -d' ' -f1 || true
  pgrep -f -a "${GWP_WATCH_MATCH}"  2>/dev/null | cut -d' ' -f1 || true
  return 0
}

gwp_family_running() {
  [[ -n "$(gwp_family_pids | grep -vx "$$" | grep . )" ]]
}

gwp_family_desc() {   # 给人看的证据行
  pgrep -f -a "${GWP_TRAIN_MATCH}" 2>/dev/null | head -2 || true
  pgrep -f -a "${GWP_TASK_MATCH}"   2>/dev/null | head -2 || true
  pgrep -f -a "${GWP_WATCH_MATCH}"  2>/dev/null | head -2 || true
  return 0
}

# launch_train.sh 用的硬闸门：GWP 相关进程存在就拒绝启动。
# 确认要越权：TAU_IGNORE_GWP_GATE=1 bash launch_train.sh ...
gwp_gate_or_die() {
  [[ "${TAU_IGNORE_GWP_GATE:-0}" == "1" ]] && {
    echo "[gate] WARN：TAU_IGNORE_GWP_GATE=1，跳过 GWP 门控" >&2; return 0; }
  if gwp_family_running; then
    echo "[gate] REFUSED：GigaWorld-Policy 还没收口，τ0-WM 不许启动。命中进程：" >&2
    gwp_family_desc >&2
    echo "[gate] 等它跑完（或人工确认越权：TAU_IGNORE_GWP_GATE=1 bash launch_train.sh ...）" >&2
    exit 75
  fi
  return 0
}
