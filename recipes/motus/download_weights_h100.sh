#!/usr/bin/env bash
# h100-103: Motus 预训练权重准备（v3: 大文件 16 路 range 并行）
# huggingface_hub 的 httpx 下载器在本机对大文件挂死（Connection refused / 0 字节停滞），
# 但 curl 直连正常；且 hf-mirror 对 motus-robotics 大文件 302 → 海外 Xet CDN，
# 单连接仅 20-200KB/s，实测 16 路 range 并行可达 15-25MB/s（25G 权重 ≈25 分钟）。
# 小文件仍用单连接 curl（dl 函数），大文件走 dl_parallel_h100.sh。
# - Wan2.2-TI2V-5B: 软链 /data1/dreamzero-pretrained/Wan2.2-TI2V-5B（VAE + umt5 T5 + tokenizer）
# - Qwen3-VL-2B-Instruct: modelscope（~4.4G）
# - Motus_Wan2_2_5B_pretrain / Motus: hf-mirror（~24.4G，modelscope 上无此仓库）
# 用法: bash /data1/motus/recipes/download_weights_h100.sh
set -uo pipefail

MOTUS_ROOT="${MOTUS_ROOT:-/data1/motus/Motus}"
PRE="${PRE:-${MOTUS_ROOT}/pretrained_models}"
LOCAL_WAN22="/data1/dreamzero-pretrained/Wan2.2-TI2V-5B"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-${HERE}/download_weights_h100.log}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

mkdir -p "${PRE}"

log "========== Motus weights (curl v2) =========="

# 1) Wan2.2: 软链复用 dreamzero 的下载
if [[ -f "${PRE}/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" ]]; then
  log "[skip] Wan2.2 already at ${PRE}/Wan2.2-TI2V-5B"
elif [[ -f "${LOCAL_WAN22}/Wan2.2_VAE.pth" ]]; then
  ln -sfn "${LOCAL_WAN22}" "${PRE}/Wan2.2-TI2V-5B"
  log "[link] ${PRE}/Wan2.2-TI2V-5B -> ${LOCAL_WAN22}"
else
  log "ERROR: ${LOCAL_WAN22} missing" >&2
  exit 1
fi

# 2) curl 下载函数：断点续传 + 重试
dl() {
  local url="$1" dest="$2" min_bytes="${3:-10}"
  local dir; dir="$(dirname "${dest}")"
  mkdir -p "${dir}"
  if [[ -f "${dest}" ]] && [[ "$(stat -c%s "${dest}")" -ge "${min_bytes}" ]]; then
    log "[skip] $(basename "${dest}") 已存在 ($(stat -c%s "${dest}") bytes)"
    return 0
  fi
  log "[dl] ${url}"
  for attempt in 1 2 3 4 5; do
    if curl -fL --retry 3 --retry-delay 5 -C - \
        --connect-timeout 15 --speed-limit 10240 --speed-time 60 \
        -o "${dest}" "${url}" 2>>"$LOG_FILE"; then
      local sz; sz="$(stat -c%s "${dest}" 2>/dev/null || echo 0)"
      if [[ "${sz}" -ge "${min_bytes}" ]]; then
        log "[ok] $(basename "${dest}") ${sz} bytes"
        return 0
      fi
      log "[warn] 大小不足: ${sz} < ${min_bytes}"
    else
      log "[warn] curl 失败 (attempt ${attempt})"
    fi
    sleep $((attempt * 10))
  done
  log "[FAIL] ${url}"
  return 1
}

MS_BASE="https://www.modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct/resolve/master"
HF_BASE="https://hf-mirror.com"

fail=0

# 3) Qwen3-VL-2B-Instruct（modelscope 快）
QWEN_DIR="${PRE}/Qwen3-VL-2B-Instruct"
for f in chat_template.json config.json configuration.json generation_config.json \
         merges.txt preprocessor_config.json README.md tokenizer_config.json \
         tokenizer.json video_preprocessor_config.json vocab.json; do
  dl "${MS_BASE}/${f}" "${QWEN_DIR}/${f}" 10 || fail=1
done
dl "${MS_BASE}/model.safetensors" "${QWEN_DIR}/model.safetensors" 3000000000 || fail=1

# 4) Motus_Wan2_2_5B_pretrain（hf-mirror，9.4G 走 16 路并行）
PRETRAIN_DIR="${PRE}/Motus_Wan2_2_5B_pretrain"
for f in .gitattributes LICENSE README.md; do
  dl "${HF_BASE}/motus-robotics/Motus_Wan2_2_5B_pretrain/resolve/main/${f}" \
     "${PRETRAIN_DIR}/${f}" 10 || fail=1
done
# 注意：HF 仓库无 config.json（架构配置），需手工补——本地
# pretrained_models/Motus_Wan2_2_5B_pretrain/config.json 有清洗版（Wan2.2 官方 config
# 去掉 _class_name/_diffusers_version 字段），scp 过去即可，否则 WanVideoModel 报
# FileNotFoundError。WanModel(**cfg) 参数：dim=3072/num_heads=24/num_layers=30/
# in_dim=48/out_dim=48/ffn_dim=14336/text_dim=4096/patch_size=[1,2,2]/model_type=ti2v。
if [[ ! -f "${PRETRAIN_DIR}/config.json" ]]; then
  log "WARN: ${PRETRAIN_DIR}/config.json missing（HF 仓库无此文件，需从本地 scp）"
fi
HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${HERE_DIR}/dl_parallel_h100.sh" \
  "${HF_BASE}/motus-robotics/Motus_Wan2_2_5B_pretrain/resolve/main/model.safetensors" \
  "${PRETRAIN_DIR}/model.safetensors" 16 8000000000 || fail=1

# 5) Motus SFT ckpt（hf-mirror）
MOTUS_DIR="${PRE}/Motus"
for f in config.json .gitattributes LICENSE README.md; do
  dl "${HF_BASE}/motus-robotics/Motus/resolve/main/${f}" "${MOTUS_DIR}/${f}" 10 || fail=1
done
# 16G mp_rank 文件走 16 路并行（单连接仅 20KB/s，Xet 海外 CDN）
dl_parallel_bin="${HERE_DIR}/dl_parallel_h100.sh"
dl_url="${HF_BASE}/motus-robotics/Motus/resolve/main/mp_rank_00_model_states.pt"
dl_dest="${MOTUS_DIR}/mp_rank_00_model_states.pt"
if [[ -f "${dl_dest}" ]] && [[ "$(stat -c%s "${dl_dest}")" -ge 10000000000 ]]; then
  log "[skip] mp_rank_00_model_states.pt 已存在"
else
  bash "${dl_parallel_bin}" "${dl_url}" "${dl_dest}" 16 10000000000 || fail=1
fi

log "========== layout =========="
du -sh "${PRE}"/* 2>/dev/null
ls -lh "${PRE}/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
  "${QWEN_DIR}/model.safetensors" \
  "${PRETRAIN_DIR}/model.safetensors" \
  "${MOTUS_DIR}/mp_rank_00_model_states.pt" 2>/dev/null || true

if [[ "${fail}" != "0" ]]; then
  log "存在失败项，请重跑本脚本（断点续传）"
  exit 1
fi
log "========== weights done =========="
