#!/bin/bash
# 在 A3 上跑：等 pretrained 下完，LAN 拷到 A2，两边 smoke。
set -euo pipefail
ROOT="${LINGBOT_ROOT:-/home/lingbot-vla}"
A2="${A2:-root@10.228.7.110}"
need=(
  "${ROOT}/pretrained/lingbot-vla-v2-6b/model-00001-of-00006.safetensors"
  "${ROOT}/pretrained/lingbot-vla-v2-6b/model-00002-of-00006.safetensors"
  "${ROOT}/pretrained/lingbot-vla-v2-6b/model-00003-of-00006.safetensors"
  "${ROOT}/pretrained/lingbot-vla-v2-6b/model-00004-of-00006.safetensors"
  "${ROOT}/pretrained/lingbot-vla-v2-6b/model-00005-of-00006.safetensors"
  "${ROOT}/pretrained/lingbot-vla-v2-6b/model-00006-of-00006.safetensors"
  "${ROOT}/pretrained/lingbot-vla-v2-6b/depth/model.pt"
  "${ROOT}/pretrained/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth"
  "${ROOT}/pretrained/moge-2-vitb-normal/model.pt"
  "${ROOT}/pretrained/qwen3-tokenizer/tokenizer.json"
)
echo "[wait] pretrained"
while true; do
  ok=1
  for f in "${need[@]}"; do
    [ -s "$f" ] || ok=0
  done
  # shards 1-5 > 4GB, shard 6 > 500MB
  sz1=$(stat -c%s "${ROOT}/pretrained/lingbot-vla-v2-6b/model-00001-of-00006.safetensors" 2>/dev/null || echo 0)
  sz6=$(stat -c%s "${ROOT}/pretrained/lingbot-vla-v2-6b/model-00006-of-00006.safetensors" 2>/dev/null || echo 0)
  if [ "$ok" = 1 ] && [ "$sz1" -gt 4000000000 ] && [ "$sz6" -gt 500000000 ]; then
    if pgrep -f download_pretrained.sh >/dev/null; then
      echo "[wait] files look complete, download still running..."
      sleep 15
      continue
    fi
    break
  fi
  echo "[wait] $(date +%H:%M:%S) du=$(du -sh ${ROOT}/pretrained | awk '{print $1}')"
  sleep 30
done
echo "[copy] A3 -> A2 LAN"
ssh -o StrictHostKeyChecking=no "$A2" "mkdir -p ${ROOT}/pretrained"
tar -C "${ROOT}/pretrained" -cf - . | ssh -o StrictHostKeyChecking=no "$A2" "tar -C ${ROOT}/pretrained -xf -"
echo "[smoke] A3"
docker exec lingbot-vla bash -lc "source ${ROOT}/env_npu.sh >/dev/null; python ${ROOT}/code/huawei/smoke_test.py" \
  || docker exec lingbot-vla bash -lc "export PYTHONPATH=${ROOT}/src/lingbot-vla-v2:${ROOT}/shim; source ${ROOT}/venv/bin/activate; python ${ROOT}/code/huawei/smoke_test.py"
echo "[smoke] A2"
ssh -o StrictHostKeyChecking=no "$A2" \
  "docker exec lingbot-vla bash -lc 'export PYTHONPATH=${ROOT}/src/lingbot-vla-v2:${ROOT}/shim; source ${ROOT}/venv/bin/activate; python ${ROOT}/code/huawei/smoke_test.py'"
echo FINISH_LAN_SMOKE_OK
