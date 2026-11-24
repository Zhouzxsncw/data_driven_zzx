#!/usr/bin/env bash
# 下载 τ0-WM 预训练权重（VAM + Action-Conditioned Video Simulator）到 /data。
# 权重源：https://huggingface.co/sii-research/tau-0-wm （apache-2.0，不 gated）
#   VAM        diffusion_pytorch_model-0000{1,2}-of-00002.bin   ≈ 11.02 G
#   simulator  simulator/diffusion_pytorch_model.bin           ≈ 11.90 G
# 合计 ≈ 22.93 G。Wan2.2-TI2V-5B 的 T5 / tokenizer / VAE 三件由 link_wan.sh 软链本机已有副本，不重复下载 32 G。
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# hf-mirror 不代理 Xet CAS（cas-server.xethub.hf.co 直连会 401），必须回退普通 HTTP 下载。
export HF_HUB_DISABLE_XET=1

TARGET="${TARGET:-/data/pretrained/tau-0-wm}"
REPO_ID="${REPO_ID:-sii-research/tau-0-wm}"

mkdir -p "${TARGET}"
echo "[tau0_wm] repo     = ${REPO_ID}"
echo "[tau0_wm] endpoint = ${HF_ENDPOINT}"
echo "[tau0_wm] target   = ${TARGET}"

hf download "${REPO_ID}" \
    --local-dir "${TARGET}" \
    --exclude ".gitattributes"

echo "[tau0_wm] 下载完成，校验体积："
du -sh "${TARGET}" "${TARGET}/simulator" 2>/dev/null || true
ls -la "${TARGET}"
