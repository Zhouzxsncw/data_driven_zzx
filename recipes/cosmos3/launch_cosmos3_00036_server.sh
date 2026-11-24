#!/usr/bin/env bash
# Cosmos3 TongBot 00036 SFT 本机服务端（端口 1236）
# 权重: h100_ckpts/cosmos3_00036/iter_000001500 + tongbot_16d_00036 quantile
#
#   bash recipes/cosmos3/launch_cosmos3_00036_server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COSMOS3_CKPT="${COSMOS3_CKPT:-/home/xk/VLA/h100_ckpts/cosmos3_00036/iter_000001500}"
export COSMOS3_PROMPT="${COSMOS3_PROMPT:-With both hands, pick up the yellow banana and the yellow lemon and put them into the black box; then take the green vegetables and place them into the blue box.}"
export COSMOS3_STEPS="${COSMOS3_STEPS:-5}"
STATS="${COSMOS3_ACTION_STATS:-/home/xk/VLA/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/tongbot_16d_00036.json}"

exec bash "$ROOT/launch_cosmos3_server.sh" --action-stats "$STATS" "$@"
