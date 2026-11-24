#!/usr/bin/env bash
# 从推理机 gpu-cigai-20 拉 00036 推理所需权重（不含基座 transformer）。
#   transformer  9.5G  checkpoint_step_2500
#   vae          2.7G  lingbot-va-base
#   text_encoder  11G  lingbot-va-base
#   tokenizer    21M
set -euo pipefail

REMOTE="${REMOTE:-gpu-cigai-20}"
DST="${DST:-/home/xk/VLA/lingbot-va-pretrained}"
INFER_DST="${INFER_DST:-/home/xk/VLA/lingbot-va-infer}"
STEP="${1:-all}"

mkdir -p \
  "$DST/checkpoint_step_2500/transformer" \
  "$DST/lingbot-va-base" \
  "$INFER_DST/example_images" \
  "$INFER_DST/results"

if [[ "$STEP" == "all" || "$STEP" == "transformer" ]]; then
  echo "[1/3] transformer checkpoint_step_2500 (infer PC SSD)"
  rsync -aH --info=progress2 \
    "${REMOTE}:/home/cigai22/lingbot-va-pretrained/checkpoint_step_2500/transformer/" \
    "${DST}/checkpoint_step_2500/transformer/"
fi

if [[ "$STEP" == "all" || "$STEP" == "base" ]]; then
  echo "[2/3] base vae/tokenizer/text_encoder/assets"
  rsync -aH --info=progress2 --exclude transformer \
    "${REMOTE}:/home/cigai22/lingbot-va-pretrained/lingbot-va-base/" \
    "${DST}/lingbot-va-base/"
fi

if [[ "$STEP" == "all" || "$STEP" == "images" ]]; then
  echo "[3/3] example images"
  rsync -aH --info=progress2 \
    "${REMOTE}:/home/cigai22/lingbot-va-infer/example_images/" \
    "${INFER_DST}/example_images/"
fi

echo done
du -sh "$DST/checkpoint_step_2500" \
  "$DST/lingbot-va-base/vae" \
  "$DST/lingbot-va-base/tokenizer" \
  "$DST/lingbot-va-base/text_encoder" \
  "$DST/lingbot-va-base/assets" \
  "$INFER_DST/example_images" 2>/dev/null || true
