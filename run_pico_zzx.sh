#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec docker run --rm --init --network host -e NVIDIA_VISIBLE_DEVICES=void \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$ROOT/vr_pub.py,dst=/workspace/vr_pub.py,readonly" \
  --entrypoint /usr/bin/python3 data-driven-zzx/pico:1.0.0 \
  -B /workspace/vr_pub.py --port 5565 "$@"
