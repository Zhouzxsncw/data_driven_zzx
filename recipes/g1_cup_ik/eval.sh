#!/usr/bin/env bash
# 等 ckpt → 先起 PolicyServer(5550) → Isaac 起来后再按剩余显存堆并行 env。
# 不杀训练。NUM_ENVS 默认 auto，不钉死 8。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
# shellcheck source=../gr00t_sonic/env.sh
source "${SCRIPT_DIR}/../gr00t_sonic/env.sh"

EXP_DIR="${EXP_DIR:-/data/gr00t_sonic_outputs/g1_cup_ik_sonic}"
POLICY_PORT="${POLICY_PORT:-5550}"
NUM_ENVS="${NUM_ENVS:-auto}"
NUM_EPISODES="${NUM_EPISODES:-auto}"
NUM_ENVS_MIN="${NUM_ENVS_MIN:-4}"
NUM_ENVS_MAX="${NUM_ENVS_MAX:-128}"
# 先够装 PolicyServer + Isaac 底 + 最少 env；装完再按剩余堆 batch
SERVER_MIB_NEED="${SERVER_MIB_NEED:-14000}"
ISAAC_BASE_MIB="${ISAAC_BASE_MIB:-8000}"
PER_ENV_MIB="${PER_ENV_MIB:-900}"
HEADROOM_MIB="${HEADROOM_MIB:-4000}"
ISAAC_ENV="${ISAAC_ENV:-${HOME}/VLA/env_isaaclab}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
# nohup 会把 TERM 设成 dumb；isaaclab.sh 的 `tabs 4` + set -e 会直接退出
export TERM=xterm-256color
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${HOME}/VLA/Isaac-GR00T/checkpoints/Cosmos-Reason2-2B}"
export PYTHONPATH="${HOME}/VLA/Isaac-GR00T:${PYTHONPATH:-}"
export GR00T_DIR="${GR00T_DIR:-${HOME}/VLA/Isaac-GR00T}"

latest_ckpt() {
    ls -d "${EXP_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1
}

gpu_free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '
}

gpu_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '
}

echo "[eval] wait ckpt under ${EXP_DIR}"
CKPT=""
for _ in $(seq 1 360); do
    CKPT="$(latest_ckpt || true)"
    if [ -n "${CKPT}" ] && [ -f "${CKPT}/config.json" ]; then
        break
    fi
    sleep 5
done
if [ -z "${CKPT}" ]; then
    echo "[eval] 没有 checkpoint" >&2
    exit 1
fi
echo "[eval] ckpt=${CKPT}"

MIN_FREE=$((SERVER_MIB_NEED + ISAAC_BASE_MIB + NUM_ENVS_MIN * PER_ENV_MIB + HEADROOM_MIB))
echo "[eval] wait GPU free>=${MIN_FREE} MiB（不杀训练；PolicyServer+Isaac 底+${NUM_ENVS_MIN} env）"
for _ in $(seq 1 720); do
    FREE="$(gpu_free_mib)"
    echo "[eval] free=${FREE} MiB"
    if [ "${FREE}" -ge "${MIN_FREE}" ]; then
        break
    fi
    sleep 10
done
FREE="$(gpu_free_mib)"
if [ "${FREE}" -lt "${MIN_FREE}" ]; then
    echo "[eval] 显存仍不够 free=${FREE} need=${MIN_FREE}" >&2
    exit 1
fi

USED_BEFORE="$(gpu_used_mib)"
if ss -ltn 2>/dev/null | grep -q ":${POLICY_PORT} "; then
    echo "[eval] 端口 ${POLICY_PORT} 已有人，复用"
else
    require_gr00t
    PY="$(gr00t_python)"
    SERVER_LOG="${EXP_DIR}/policy_server.log"
    setsid nohup env \
        GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH}" \
        PYTHONPATH="${HOME}/VLA/Isaac-GR00T" \
        "${PY}" "${GR00T_DIR}/gr00t/eval/run_gr00t_server.py" \
            --model-path "${CKPT}" \
            --embodiment-tag UNITREE_G1_SONIC \
            --device cuda:0 \
            --host 0.0.0.0 \
            --port "${POLICY_PORT}" \
        >> "${SERVER_LOG}" 2>&1 < /dev/null &
    echo "[eval] PolicyServer pid=$! log=${SERVER_LOG}"
    echo "[eval] wait PolicyServer 占显存"
    for _ in $(seq 1 90); do
        USED_NOW="$(gpu_used_mib)"
        if [ "$((USED_NOW - USED_BEFORE))" -ge 5000 ]; then
            break
        fi
        sleep 2
    done
fi

# 0 = Python 在 Isaac AppLauncher 起来之后按剩余显存再堆，不在 shell 里钉死
if [ "${NUM_ENVS}" = "auto" ]; then
    NUM_ENVS=0
fi
if [ "${NUM_EPISODES}" = "auto" ]; then
    NUM_EPISODES=0
fi

OUT="${EXP_DIR}/eval_$(basename "${CKPT}")"
mkdir -p "${OUT}"
{
    echo "ckpt=${CKPT}"
    echo "num_envs=${NUM_ENVS}"
    echo "num_episodes=${NUM_EPISODES}"
    echo "gpu_free_mib=$(gpu_free_mib)"
    echo "gpu_used_mib=$(gpu_used_mib)"
} > "${OUT}/eval_launch.txt"
nvidia-smi >> "${OUT}/eval_launch.txt" || true

export CONDA_PREFIX="${ISAAC_ENV}"
export PATH="${ISAAC_ENV}/bin:${PATH}"
export PER_ENV_MIB ISAAC_BASE_MIB HEADROOM_MIB NUM_ENVS_MIN NUM_ENVS_MAX
# 不要走 isaaclab.sh：`tabs 4` + set -e 在 nohup 下会直接退出
cd "${ISAACLAB_DIR}"
exec "${ISAAC_ENV}/bin/python" "${SCRIPT_DIR}/eval_gpu_policy.py" \
    --headless \
    --device cuda:0 \
    --num_envs "${NUM_ENVS}" \
    --num_episodes "${NUM_EPISODES}" \
    --policy_port "${POLICY_PORT}" \
    --out_dir "${OUT}" \
    --ckpt_name "$(basename "${CKPT}")" \
    --lockstep
