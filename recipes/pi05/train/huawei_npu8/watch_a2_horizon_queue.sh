#!/bin/bash
# A2 上 h10 正常结束后再开 h30，h30 正常结束后再开 h90。崩溃则停。
# 在宿主机上跑：nohup bash watch_a2_horizon_queue.sh >/home/pi05/ckpt_on_software/watch_a2_horizon.log 2>&1 &
set -euo pipefail

chain() {
  local yaml="$1"
  local log="$2"
  local next="$3"
  echo "$(date -Is) wait until ${yaml} has started"
  local seen=0
  for _ in $(seq 1 120); do
    if docker exec pi05 pgrep -f "$yaml" >/dev/null; then
      seen=1
      break
    fi
    sleep 10
  done
  if [ "$seen" != 1 ]; then
    echo "$(date -Is) ${yaml} never started; stop"
    exit 1
  fi
  echo "$(date -Is) ${yaml} is up, waiting for it to exit"
  while docker exec pi05 pgrep -f "$yaml" >/dev/null; do
    sleep 60
  done
  echo "$(date -Is) ${yaml} exited"
  if [ ! -s "$log" ]; then
    echo "$(date -Is) missing log $log; not starting ${next}"
    exit 1
  fi
  if tail -n 80 "$log" | grep -q 'Traceback'; then
    echo "$(date -Is) ${yaml} ended with Traceback; not starting ${next}"
    exit 1
  fi
  echo "$(date -Is) starting ${next}"
  docker exec -d -w /home/pi05/openpi_src pi05 bash "$next"
}

chain configs/train_pi05_a2_h10.yaml \
  /home/pi05/ckpt_on_software/train_mix_a2_h10.log \
  start_train_npu8_a2_h30.sh
chain configs/train_pi05_a2_h30.yaml \
  /home/pi05/ckpt_on_software/train_mix_a2_h30.log \
  start_train_npu8_a2_h90.sh
echo "$(date -Is) h90 launched; queue watcher done"
