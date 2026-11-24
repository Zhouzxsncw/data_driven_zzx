#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE="$ROOT/pico/pc_service"
if pgrep -f -- '(^|/)RoboticsServiceProcess([[:space:]]|$)' >/dev/null; then
  echo "已有 RoboticsServiceProcess 在运行，保持现有服务。" >&2
  exit 1
fi

listeners="$(ss -H -ltn)"
if awk '$4 ~ /:(60061|63901)$/ {busy=1} END {exit !busy}' <<< "$listeners"; then
  echo "PC Service 端口 60061 或 63901 已占用，保持现有服务。" >&2
  exit 1
fi

STATE="$ROOT/pico/state"
mkdir -p "$STATE/config" "$STATE/data" "$STATE/cache"
exec docker run --rm --init --network host -e NVIDIA_VISIBLE_DEVICES=void \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$SERVICE,dst=/opt/apps/roboticsservice,readonly" \
  --mount "type=bind,src=$STATE,dst=/pico-state" \
  --workdir /opt/apps/roboticsservice \
  -e XDG_CONFIG_HOME=/pico-state/config \
  -e XDG_DATA_HOME=/pico-state/data \
  -e XDG_CACHE_HOME=/pico-state/cache \
  -e LD_LIBRARY_PATH=/opt/apps/roboticsservice:/opt/apps/roboticsservice/lib:/opt/apps/roboticsservice/SDK/x64 \
  -e QT_PLUGIN_PATH=/opt/apps/roboticsservice/plugins \
  --entrypoint /opt/apps/roboticsservice/RoboticsServiceProcess data-driven-zzx/pico:1.0.0 "$@"
