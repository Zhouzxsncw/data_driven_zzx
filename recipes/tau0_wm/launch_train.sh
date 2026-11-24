#!/usr/bin/env bash
# 启动 τ0-WM 在 TongBot 00036 上的后训练。
#
#   bash launch_train.sh                  # 正式跑（读 configs/tau_model/posttrain_tongbot_00036.yaml）
#   bash launch_train.sh --smoke          # 冒烟：bs=1 / 20 step / 不存 ckpt 不做 val
#   bash launch_train.sh -o batch_size=4 -o gradient_checkpointing=false
#   CONFIG=/path/x.yaml NGPU=4 bash launch_train.sh
#
# GWP 在跑时会直接 REFUSED（exit 75）；人工确认越权才用 TAU_IGNORE_GWP_GATE=1。
#
# 官方 scripts/train.sh 会把 nproc_per_node 写成「物理卡总数」，不认 CUDA_VISIBLE_DEVICES；
# 这里自己数卡，并且支持用 -o 临时覆盖 YAML 字段（生成的 YAML 落到运行目录留档，不污染真源）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 硬闸门（用户要求）：GigaWorld-Policy 没收口就不许起 τ0-WM。
# 判定逻辑与 h100/relay_watch.sh 共用，见 h100/gwp_gate.sh 顶部注释。
# shellcheck disable=SC1091
source "${HERE}/h100/gwp_gate.sh"
gwp_gate_or_die
TAU_ROOT="${TAU_ROOT:-/home/xk/VLA/tau-0-wm}"
# 换承载机只改这两个：h100 上 source h100/env_h100.sh 会把它设成 venv 与 EE 配置
TAU_ENV_SH="${TAU_ENV_SH:-${HERE}/env.sh}"
TAU_PY="${TAU_PY:-${HOME}/miniconda3/envs/tau0wm/bin/python}"
CONFIG="${CONFIG:-${HERE}/configs/tau_model/posttrain_tongbot_00036_ee.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG}" .yaml)}"

# 真源在 recipe 目录，repo 只认自己的相对路径 → 先同步
bash "${HERE}/sync_configs.sh" >/dev/null
REPO_CONFIG="configs/tau_model/$(basename "${CONFIG}")"

SMOKE=0
declare -a OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    -o|--override) OVERRIDES+=("$2"); shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

if [[ ${SMOKE} -eq 1 ]]; then
  OVERRIDES+=(batch_size=1 train_steps=20 steps_to_save=100000000 steps_to_val=100000000
             steps_to_log=1 dataloader_num_workers=2 gradient_checkpointing=true
             "sub_folder=_smoke_${RUN_NAME}")
fi

RUNS_ROOT="${TAU_RUNS:-/data/tau0_wm_runs}"
GEN_DIR="${RUNS_ROOT}/_generated"
mkdir -p "${GEN_DIR}"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  APPLIED="${GEN_DIR}/${RUN_NAME}$(date '+_%H%M%S').yaml"
  TAU_SRC="${CONFIG}" TAU_DST="${APPLIED}" "${TAU_PY}" - "${OVERRIDES[@]}" <<'PY'
import os, sys
import yaml

src, dst = os.environ["TAU_SRC"], os.environ["TAU_DST"]
cfg = yaml.safe_load(open(src))

def cast(old, raw):
    if isinstance(old, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(old, int) and not isinstance(old, bool):
        return int(float(raw))
    if isinstance(old, float):
        return float(raw)
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw

for kv in sys.argv[1:]:
    path, raw = kv.split("=", 1)
    node = cfg
    for key in path.split(".")[:-1]:
        node = node[key]
    leaf = path.split(".")[-1]
    node[leaf] = cast(node.get(leaf), raw)
    print(f"[launch] override {path} = {node[leaf]!r}")

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(f"[launch] 生成 YAML → {dst}")
PY
  # 用生成的副本，仍走 repo 相对路径加载 data YAML
  cp -f "${APPLIED}" "${TAU_ROOT}/${REPO_CONFIG}.run"
  REPO_CONFIG="${REPO_CONFIG}.run"
else
  cp -f "${CONFIG}" "${TAU_ROOT}/${REPO_CONFIG}"
fi

# 数卡：优先认 CUDA_VISIBLE_DEVICES
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
else
  NGPU=$(nvidia-smi --list-gpus | wc -l)
fi

OUT_DIR="${RUNS_ROOT}/tongbot_00036"
mkdir -p "${OUT_DIR}"
# 注意别写成 LOG="...$( [[ ${SMOKE} -eq 1 ]] && echo '_smoke').log"：
# 非冒烟时替换串返回 1，简单命令的赋值会把赋值自身的退出码取成替换的退出码，
# 头顶 `set -e` 直接静默 exit 1 —— 冒烟能过、正式必死（h100 bash 5.1.16 实测 rc=1 零输出）。
if [[ ${SMOKE} -eq 1 ]]; then LOG_SUFFIX='_smoke'; else LOG_SUFFIX=''; fi
LOG="${GEN_DIR}/${RUN_NAME}${LOG_SUFFIX}.log"

echo "[launch] TAU_ROOT=${TAU_ROOT}  config=${REPO_CONFIG}  NGPU=${NGPU}"
echo "[launch] log=${LOG}"

cd "${TAU_ROOT}"
# shellcheck disable=SC1091
source "${TAU_ENV_SH}" >/dev/null

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

# h100 的 uv venv 里 torch 装好了但没生成 torchrun 入口脚本（实测 command not found），
# 所以优先用 PATH 上的 torchrun，找不到就退回等价的 `python -m torch.distributed.run`。
# 用数组而不是裸字符串，避免 `set -u` 下多词命令被错误分词。
if command -v torchrun >/dev/null 2>&1; then
  RUNNER=(torchrun)
else
  echo "[launch] PATH 上没有 torchrun，改用 ${TAU_PY} -m torch.distributed.run"
  RUNNER=("${TAU_PY}" -m torch.distributed.run)
fi

"${RUNNER[@]}" --nnodes=1 --nproc_per_node="${NGPU}" --node_rank=0 \
    main.py \
    --config_file "${REPO_CONFIG}" \
    --runner_class_path runner/posttrain.py \
    --runner_class Trainer \
    2>&1 | tee "${LOG}"
