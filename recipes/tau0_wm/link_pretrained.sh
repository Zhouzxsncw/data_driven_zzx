#!/usr/bin/env bash
# 把本机已有的 Wan2.2-TI2V-5B 三件（T5 权重 / tokenizer / VAE）软链成 τ0-WM 配置里的稳定路径。
# τ0-WM 用的是 Wan 原生（非 diffusers）布局，正好和 dreamzero_staging 里那份一致，
# 不必再花 32 G 去下一遍。
set -euo pipefail

SRC="${SRC:-/home/xk/VLA/dreamzero_staging/pretrained/Wan2.2-TI2V-5B}"
DST="${DST:-/data/pretrained/Wan2.2-TI2V-5B}"

echo "[tau0_wm] SRC=${SRC}"
echo "[tau0_wm] DST=${DST}"

[[ -d "${SRC}" ]] || { echo "找不到 Wan2.2-TI2V-5B：${SRC}" >&2; exit 1; }

mkdir -p "${DST}/google"

for f in models_t5_umt5-xxl-enc-bf16.pth Wan2.2_VAE.pth; do
  [[ -e "${SRC}/${f}" ]] || { echo "缺 ${SRC}/${f}" >&2; exit 1; }
  ln -sfn "${SRC}/${f}" "${DST}/${f}"
done

[[ -d "${SRC}/google/umt5-xxl" ]] || { echo "缺 ${SRC}/google/umt5-xxl" >&2; exit 1; }
ln -sfn "${SRC}/google/umt5-xxl" "${DST}/google/umt5-xxl"

echo "[tau0_wm] 校验（软链指向的真实体积）："
ls -laL "${DST}/${DST##*/}" 2>/dev/null || true
for f in "${DST}/models_t5_umt5-xxl-enc-bf16.pth" "${DST}/Wan2.2_VAE.pth"; do
  printf "  %-58s %s\n" "$f" "$(du -shL "$f" | cut -f1)"
done
echo "  ${DST}/google/umt5-xxl:"
ls "${DST}/google/umt5-xxl/" | sed 's/^/    /'
