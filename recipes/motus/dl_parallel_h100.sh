#!/usr/bin/env bash
# h100-103: 大文件多连接 range 并行下载（对抗 Xet 海外 CDN 单连接慢速）
# 背景: hf-mirror 对 motus-robotics 大文件 302 → cas-bridge.xethub.hf.co（海外 Xet CDN），
#       单连接仅 20-200KB/s；实测 4 路 range 并行合计 ~830KB/s，多连接近线性提升。
# gateway 内网通道仅 ~0.76MB/s（00091 上传 5.2G 用了 7135s），本地推送不可行。
# 用法: bash dl_parallel_h100.sh URL DEST [N_PARTS] [MIN_BYTES]
#       分片写 DEST.partNNN，全部完成后按数字序拼接并校验总大小，支持断点续跑。
set -uo pipefail

URL="${1:?url}"
DEST="${2:?dest}"
N="${3:-16}"
MIN_BYTES="${4:-1000000000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-${HERE}/dl_parallel.log}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

if [[ -f "${DEST}" ]] && [[ "$(stat -c%s "${DEST}")" -ge "${MIN_BYTES}" ]]; then
  log "[skip] ${DEST} complete ($(stat -c%s "${DEST}") bytes)"
  exit 0
fi
mkdir -p "$(dirname "${DEST}")"

# 文件总大小（跟随 302 后取 content-length，重试）
SIZE=""
for _ in 1 2 3 4 5; do
  SIZE="$(curl -sIL --connect-timeout 15 "${URL}" | tr -d '\r' | grep -i '^content-length:' | tail -1 | awk '{print $2}')"
  [[ -n "${SIZE}" ]] && break
  sleep 5
done
if [[ -z "${SIZE}" ]] || ! [[ "${SIZE}" =~ ^[0-9]+$ ]]; then
  log "[FAIL] size unknown for ${URL}"
  exit 1
fi
log "[start] $(basename "${DEST}") size=${SIZE} parts=${N} url=${URL}"

CHUNK=$(( (SIZE + N - 1) / N ))

# 单分片下载：追加模式断点续传（从已有字节后继续 range），慢速自动重连
dl_part() {
  local idx="$1" start="$2" end="$3"
  local want=$(( end - start + 1 ))
  local part="${DEST}.part$(printf '%03d' "${idx}")"
  local have=0
  [[ -f "${part}" ]] && have="$(stat -c%s "${part}")"
  local round=0
  while [[ "${have}" -lt "${want}" ]] && [[ "${round}" -lt 15 ]]; do
    round=$((round + 1))
    local from=$(( start + have ))
    curl -fsL --connect-timeout 20 --speed-limit 4096 --speed-time 60 \
      -r "${from}-${end}" -o - "${URL}" >> "${part}" 2>>"$LOG_FILE" || true
    have="$(stat -c%s "${part}" 2>/dev/null || echo 0)"
  done
  if [[ "${have}" -eq "${want}" ]]; then
    return 0
  fi
  log "[part $(printf '%03d' "${idx}")] FAILED have=${have}/${want} after ${round} rounds"
  return 1
}

pids=()
fail=0
for i in $(seq 0 $((N - 1))); do
  start=$(( i * CHUNK ))
  end=$(( start + CHUNK - 1 ))
  (( end >= SIZE )) && end=$(( SIZE - 1 ))
  dl_part "${i}" "${start}" "${end}" &
  pids+=($!)
done
for p in "${pids[@]}"; do
  wait "${p}" || fail=1
done

if [[ "${fail}" -ne 0 ]]; then
  log "[FAIL] incomplete parts remain; rerun this script to resume"
  exit 1
fi

# 按数字序拼接（不可 glob：part010 字典序在 part002 前）
: > "${DEST}.cat"
for i in $(seq 0 $((N - 1))); do
  cat "${DEST}.part$(printf '%03d' "${i}")" >> "${DEST}.cat"
done
total="$(stat -c%s "${DEST}.cat")"
if [[ "${total}" -eq "${SIZE}" ]]; then
  mv "${DEST}.cat" "${DEST}"
  rm -f "${DEST}".part*
  log "[done] ${DEST} ${total} bytes"
  exit 0
else
  log "[FAIL] concat size ${total} != ${SIZE}"
  exit 1
fi
