#!/usr/bin/env bash
# Cosmos3 TongBot 00091 SFT 本机服务端（端口 1236）
# 权重: h100_ckpts/cosmos3_00091/iter_000001500 + tongbot_16d_00091 quantile
# （iter_000003000 本机未拉完整，不能用）
#
#   bash recipes/cosmos3/launch_cosmos3_00091_server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COSMOS3_CKPT="${COSMOS3_CKPT:-/home/xk/VLA/h100_ckpts/cosmos3_00091/iter_000001500}"
export COSMOS3_PROMPT="${COSMOS3_PROMPT:-Using both hands, pull open the drawer with the blue handle using your left hand, take out the white medicine box with your right hand, place it into the green drawer, and finally close the drawer.}"
export COSMOS3_STEPS="${COSMOS3_STEPS:-10}"
STATS="${COSMOS3_ACTION_STATS:-/home/xk/VLA/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/tongbot_16d_00091.json}"

exec bash "$ROOT/launch_cosmos3_server.sh" --action-stats "$STATS" "$@"
