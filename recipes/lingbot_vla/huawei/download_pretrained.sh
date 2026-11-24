#!/bin/bash
# 在华为机器本地下载 LingBot-VLA 预训练（hf-mirror），可断点续传。
# 不走 UniVPN 从 gpu-master 推 27G。
set -euo pipefail
ROOT="${LINGBOT_ROOT:-/home/lingbot-vla}"
PRE="${ROOT}/pretrained"
MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
mkdir -p "${PRE}/lingbot-vla-v2-6b/depth" "${PRE}/lingbot-vla-v2-6b/dino_video" \
         "${PRE}/moge-2-vitb-normal" "${PRE}/qwen3-tokenizer" "${PRE}/Qwen3-VL-4B-Instruct"

fetch() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  echo "[get] $dest"
  wget -c --timeout=30 --tries=0 --retry-connrefused -O "$dest" "$url"
}

# 并行度默认 2，避免把华为出口打满。
CONC="${CONC:-2}"
fifo=$(mktemp -u)
mkfifo "$fifo"
# shellcheck disable=SC2094
exec 3<>"$fifo"
rm -f "$fifo"
for _ in $(seq 1 "$CONC"); do echo >&3; done

run() {
  read -r -u 3
  {
    fetch "$1" "$2" || echo "[FAIL] $2" >&2
    echo >&3
  } &
}

VLA="${MIRROR}/robbyant/lingbot-vla-v2-6b/resolve/main"
for i in 1 2 3 4 5 6; do
  f=$(printf 'model-%05d-of-00006.safetensors' "$i")
  run "${VLA}/${f}" "${PRE}/lingbot-vla-v2-6b/${f}"
done
for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json preprocessor_config.json video_preprocessor_config.json \
         added_tokens.json vocab.json README.md; do
  run "${VLA}/${f}" "${PRE}/lingbot-vla-v2-6b/${f}"
done
run "${VLA}/depth/model.pt" "${PRE}/lingbot-vla-v2-6b/depth/model.pt"
run "${VLA}/dino_video/teacher_step_10000.pth" "${PRE}/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth"
run "${VLA}/dino_video/config.yaml" "${PRE}/lingbot-vla-v2-6b/dino_video/config.yaml"

run "${MIRROR}/Ruicheng/moge-2-vitb-normal/resolve/main/model.pt" \
    "${PRE}/moge-2-vitb-normal/model.pt"

Q="${MIRROR}/Qwen/Qwen3-VL-4B-Instruct/resolve/main"
for f in tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt \
         added_tokens.json preprocessor_config.json video_preprocessor_config.json \
         config.json generation_config.json chat_template.json \
         model.safetensors.index.json \
         model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  run "${Q}/${f}" "${PRE}/Qwen3-VL-4B-Instruct/${f}"
done

wait
exec 3>&-

# tokenizer 目录：优先复制 Qwen 的 tokenizer 文件
if [ ! -f "${PRE}/qwen3-tokenizer/tokenizer.json" ]; then
  mkdir -p "${PRE}/qwen3-tokenizer"
  cp -n "${PRE}/Qwen3-VL-4B-Instruct/"*.json \
        "${PRE}/Qwen3-VL-4B-Instruct/"tokenizer.json \
        "${PRE}/Qwen3-VL-4B-Instruct/"vocab.json \
        "${PRE}/Qwen3-VL-4B-Instruct/"merges.txt \
        "${PRE}/qwen3-tokenizer/" 2>/dev/null || true
fi
echo "[OK] download pass (check sizes below)"
du -sh "${PRE}" "${PRE}"/* 2>/dev/null
ls -lh "${PRE}/lingbot-vla-v2-6b/"*.safetensors 2>/dev/null
