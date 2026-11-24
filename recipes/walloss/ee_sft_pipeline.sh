#!/usr/bin/env bash
# TongBot RM75 → WALL-OSS-0.5 EE SFT pipeline
#
# 机械臂: 睿尔曼 RealMan RM75 双臂（16D joint）。
# 阶段: convert(joint→EE) → 写 YAML → relative norm → SFT
#
# 用法:
#   bash ee_sft_pipeline.sh --src /path/to/lerobot_v21
#   bash ee_sft_pipeline.sh --src /path/to/lerobot_v21 --until norm
#   bash ee_sft_pipeline.sh --src /data/wall-oss/datasets/00086_rm75_ee --until train --nproc 1
#
# 已转过 EE 的目录（有 meta/joint_to_ee.json）会自动跳过 convert。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="${SCRIPT_DIR}"
CONVERT_PY="${SCRIPT_DIR}/convert_tongbot_joint_to_ee.py"
TEMPLATE="${SETUP_DIR}/train/sft_tongbot_ee.template.yml"

SRC=""
DST=""
NAME=""
UNTIL="train"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29617}"
SDK_ROOT="${SDK_ROOT:-/home/xk/RM_API}"
LIMIT_EPISODES=""
OVERWRITE=0
DRY_RUN=0
RES="${RES:-256}"
BS="${BS:-1}"
ACCUM="${ACCUM:-4}"
NUM_EPOCH="${NUM_EPOCH:-20}"
MAX_STEPS="${MAX_STEPS:-20000}"
WARMUP="${WARMUP:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
FORCE_TYPE="${FORCE_TYPE:-RM_MODEL_RM_B_E}"
J7_OFFSET="${J7_OFFSET:-120}"
CONVERT_WORKERS="${CONVERT_WORKERS:-1}"
ACTION_TOKENIZER_TYPE="${ACTION_TOKENIZER_TYPE:-fast}"
ACTION_TOKENIZER_PATH="${ACTION_TOKENIZER_PATH:-/data/wall-oss/pretrained/fast}"
USE_FAST_TOKENIZER="${USE_FAST_TOKENIZER:-true}"

usage() {
  cat <<'EOF'
TongBot RM75 joint → EE → WALL-OSS-0.5 SFT

  bash ee_sft_pipeline.sh --src <lerobot_or_ee_dir> [options]

必选:
  --src DIR              原始 16D joint 数据集，或已转好的 *_rm75_ee 目录

可选:
  --dst DIR              EE 输出目录（默认 <data_root>/<name>_rm75_ee）
  --name TAG             跑次名（默认从 src 目录名取 5 位编号，如 00086）
  --until convert|norm|train
                         停在哪一步（默认 train）
  --nproc N              训练卡数（默认 1；江算 16 卡设 16）
  --overwrite            已有 EE 数据也重转
  --limit-episodes N     只转前 N 条（冒烟）
  --sdk-root DIR         RealMan SDK（默认 /home/xk/RM_API）
  --dry-run              只打印命令
  --res 256              训练图像边长（共享 GPU 默认 256；独占卡可 448）
  --bs / --accum / --max-steps / --num-epoch / --warmup

环境（自动探测）:
  本机     source /data/wall-oss/env.sh
  江算容器 source /workspace/venv + PPU envsetup

示例:
  # 本机: 00086 全流程单卡
  bash recipes/walloss/ee_sft_pipeline.sh \
    --src /home/xk/dataset/00086_20260827_RM_01_YJ_ZL_21

  # 只准备数据，不开训
  bash recipes/walloss/ee_sft_pipeline.sh --src /path/to/00073_... --until norm

  # 江算 16 卡（EE 数据需已放到容器可见路径）
  bash /workspace/code/ee_sft_pipeline.sh \
    --src /data/datasets/00073_rm75_ee --until train --nproc 16
EOF
}

log() { printf '[pipeline] %s\n' "$*"; }
die() { printf '[pipeline] ERROR: %s\n' "$*" >&2; exit 1; }

run() {
  log "+ $*"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --src) SRC="$2"; shift 2 ;;
    --dst) DST="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --until) UNTIL="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --sdk-root) SDK_ROOT="$2"; shift 2 ;;
    --limit-episodes) LIMIT_EPISODES="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --res) RES="$2"; shift 2 ;;
    --bs) BS="$2"; shift 2 ;;
    --accum) ACCUM="$2"; shift 2 ;;
    --num-epoch) NUM_EPOCH="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --convert-workers) CONVERT_WORKERS="$2"; shift 2 ;;
    --action-tokenizer-type) ACTION_TOKENIZER_TYPE="$2"; shift 2 ;;
    --action-tokenizer-path) ACTION_TOKENIZER_PATH="$2"; shift 2 ;;
    --use-fast-tokenizer) USE_FAST_TOKENIZER="$2"; shift 2 ;;
    *) die "未知参数: $1（--help 看用法）" ;;
  esac
done

[[ -n "${SRC}" ]] || { usage >&2; die "需要 --src"; }
[[ "${UNTIL}" =~ ^(convert|norm|train|infer-test)$ ]] || die "--until 只能是 convert|norm|train|infer-test"
SRC="$(cd "${SRC}" && pwd)"
[[ -f "${SRC}/meta/info.json" ]] || die "不是 LeRobot 根目录（缺 meta/info.json）: ${SRC}"
[[ -f "${CONVERT_PY}" ]] || die "找不到 ${CONVERT_PY}"
[[ -f "${TEMPLATE}" ]] || die "找不到 ${TEMPLATE}"

if [[ -z "${NAME}" ]]; then
  base="$(basename "${SRC}")"
  if [[ "${base}" =~ ^([0-9]{5}) ]]; then
    NAME="${BASH_REMATCH[1]}"
  else
    NAME="${base}"
  fi
  NAME="${NAME%_rm75_ee}"
fi

if [[ -d /data/wall-oss ]]; then
  DATA_ROOT="${DATA_ROOT:-/data/wall-oss}"
elif [[ -d /nvme2n1/wall-oss ]]; then
  DATA_ROOT="${DATA_ROOT:-/nvme2n1/wall-oss}"
elif [[ -d /workspace ]]; then
  DATA_ROOT="${DATA_ROOT:-/workspace}"
else
  DATA_ROOT="${DATA_ROOT:-${HOME}/wall-oss-work}"
fi

if [[ -z "${DST}" ]]; then
  if [[ -f "${SRC}/meta/joint_to_ee.json" ]]; then
    DST="${SRC}"
  else
    DST="${DATA_ROOT}/datasets/${NAME}_rm75_ee"
  fi
fi

PRETRAINED="${PRETRAINED:-${DATA_ROOT}/pretrained/wall-oss-0.5}"
PROCESSOR="${PROCESSOR:-${DATA_ROOT}/pretrained/Qwen2.5-VL-3B-Instruct}"
RESUME="${RESUME:-${PRETRAINED}/model.safetensors}"
WALLX="${WALLX:-${DATA_ROOT}/src/wall-x}"
SAVE_PATH="${SAVE_PATH:-${DATA_ROOT}/outputs/walloss05_tongbot_ee_${NAME}}"
NORM_PATH="${DST}/meta/walloss_norm_stats.json"
CFG_PATH="${DST}/sft.yml"
LOG_NAME="walloss05_tongbot_ee_${NAME}"

stage_rank() {
  case "$1" in
    convert) echo 1 ;;
    infer-test) echo 1 ;;
    norm) echo 2 ;;
    train) echo 3 ;;
    *) echo 0 ;;
  esac
}
WANT="$(stage_rank "${UNTIL}")"

src_is_ee=0
if [[ -f "${SRC}/meta/joint_to_ee.json" ]]; then
  src_is_ee=1
fi

need_convert=1
if [[ "${src_is_ee}" -eq 1 && "${DST}" == "${SRC}" ]]; then
  need_convert=0
fi
if [[ -f "${DST}/meta/joint_to_ee.json" && "${OVERWRITE}" -eq 0 ]]; then
  need_convert=0
fi
# 传入的目录已经带 action.ee 时，就在这份数据上写 yaml / norm，不再 FK。
if [[ "${OVERWRITE}" -eq 0 ]]; then
  if python3 -c "import json,sys; d=json.load(open('${SRC}/meta/info.json')); sys.exit(0 if 'action.ee' in (d.get('features') or {}) else 1)"; then
    need_convert=0
    DST="${SRC}"
  fi
fi
NORM_PATH="${DST}/meta/walloss_norm_stats.json"
CFG_PATH="${DST}/sft.yml"

log "robot     RealMan RM75 dual-arm (16D joint → 20D EE, pad to 26)"
log "src       ${SRC}"
log "ee dst    ${DST}"
log "name      ${NAME}"
log "until     ${UNTIL}"
log "nproc     ${NPROC}"
log "cfg       ${CFG_PATH}"
log "ckpt      ${SAVE_PATH}"

# 单卡共享 GPU：DDP+bf16（FSDP 把参数留 fp32，Adam 状态会 OOM）
# 多卡：FSDP，不 offload
if [[ "${NPROC}" -gt 1 ]]; then
  USE_FSDP="${USE_FSDP:-true}"
  FSDP_CPU_OFFLOAD="${FSDP_CPU_OFFLOAD:-false}"
else
  USE_FSDP="${USE_FSDP:-false}"
  FSDP_CPU_OFFLOAD="${FSDP_CPU_OFFLOAD:-false}"
fi
log "fsdp      ${USE_FSDP}  cpu_offload=${FSDP_CPU_OFFLOAD}"

activate_walloss() {
  if [[ -f /data/wall-oss/env.sh ]]; then
    # shellcheck disable=SC1091
    source /data/wall-oss/env.sh
  elif [[ -f /nvme2n1/wall-oss/env.sh ]]; then
    # shellcheck disable=SC1091
    source /nvme2n1/wall-oss/env.sh
  elif [[ -f /workspace/venv/bin/activate ]]; then
    set +u
    # shellcheck disable=SC1091
    [[ -f /usr/local/PPU_SDK/envsetup.sh ]] && source /usr/local/PPU_SDK/envsetup.sh
    # shellcheck disable=SC1091
    source /workspace/venv/bin/activate
    set -u
    export ROOT="${ROOT:-/workspace}"
    export WALLX_SKIP_CUDA_EXT="${WALLX_SKIP_CUDA_EXT:-1}"
  else
    die "找不到 WALL-OSS 环境（/data/wall-oss/env.sh 或 /workspace/venv）"
  fi
  command -v python >/dev/null || die "activate 后没有 python"
  set -euo pipefail
}

render_yaml() {
  python3 - "${TEMPLATE}" "${CFG_PATH}" \
    "${DST}" "${NORM_PATH}" "${LOG_NAME}" "${SAVE_PATH}" \
    "${PRETRAINED}" "${PROCESSOR}" "${RESUME}" \
    "${RES}" "${BS}" "${ACCUM}" "${NUM_EPOCH}" "${MAX_STEPS}" "${WARMUP}" \
    "${NUM_WORKERS}" "${SAVE_INTERVAL}" "${VAL_INTERVAL}" \
    "${USE_FSDP}" "${FSDP_CPU_OFFLOAD}" \
    "${ACTION_TOKENIZER_TYPE}" "${ACTION_TOKENIZER_PATH}" "${USE_FAST_TOKENIZER}" <<'PY'
import json
import sys
from pathlib import Path
tmpl, out, ee_dst, norm, log_name, save, pretrained, processor, resume, res, bs, accum, epoch, steps, warmup, workers, save_iv, val_iv, use_fsdp, cpu_offload, tok_type, tok_path, use_fast = sys.argv[1:]
text = Path(tmpl).read_text()
info = json.loads((Path(ee_dst) / "meta" / "info.json").read_text())
feats = info.get("features") or {}

def pick(*cands: str) -> str:
    for cand in cands:
        if cand in feats:
            return cand
    raise SystemExit(f"数据集缺少相机键 {cands}，已有: {sorted(feats)}")

text = text.replace("observation.images.front:", pick("observation.images.front", "observation.images.cam_mid") + ":")
text = text.replace("observation.images.left:", pick("observation.images.left", "observation.images.cam_left") + ":")
text = text.replace("observation.images.right:", pick("observation.images.right", "observation.images.cam_right") + ":")
repl = {
    "{{EE_DST}}": ee_dst,
    "{{NORM_PATH}}": norm,
    "{{LOG_NAME}}": log_name,
    "{{SAVE_PATH}}": save,
    "{{PRETRAINED}}": pretrained,
    "{{PROCESSOR}}": processor,
    "{{RESUME}}": resume,
    "{{RES}}": res,
    "{{BS}}": bs,
    "{{ACCUM}}": accum,
    "{{NUM_EPOCH}}": epoch,
    "{{MAX_STEPS}}": steps,
    "{{WARMUP}}": warmup,
    "{{NUM_WORKERS}}": workers,
    "{{SAVE_INTERVAL}}": save_iv,
    "{{VAL_INTERVAL}}": val_iv,
    "{{USE_FSDP}}": use_fsdp,
    "{{FSDP_CPU_OFFLOAD}}": cpu_offload,
    "{{ACTION_TOKENIZER_TYPE}}": tok_type,
    "{{ACTION_TOKENIZER_PATH}}": tok_path,
    "{{USE_FAST_TOKENIZER}}": use_fast,
}
for k, v in repl.items():
    text = text.replace(k, v)
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(text)
print(out)
PY
}

# --- 1. convert ---
if [[ "${WANT}" -ge 1 ]]; then
  if [[ "${need_convert}" -eq 0 ]]; then
    log "convert   skip（已有 ${DST}/meta/joint_to_ee.json）"
  else
    [[ -d "${SDK_ROOT}" ]] || die "没有 RealMan SDK: ${SDK_ROOT}（convert 必须在有 RM_API 的机器上跑）"
    mkdir -p "$(dirname "${DST}")"
    convert_cmd=(
      python3 "${CONVERT_PY}"
      --src "${SRC}"
      --dst "${DST}"
      --sdk-root "${SDK_ROOT}"
      --force-type "${FORCE_TYPE}"
      --j7-offset-deg "${J7_OFFSET}"
    )
    [[ -n "${LIMIT_EPISODES}" ]] && convert_cmd+=(--limit-episodes "${LIMIT_EPISODES}")
    [[ "${OVERWRITE}" -eq 1 ]] && convert_cmd+=(--overwrite)
    convert_cmd+=(--workers "${CONVERT_WORKERS}")
    export PYTHONPATH="${SDK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    run "${convert_cmd[@]}"
  fi
fi

if [[ "${UNTIL}" == "infer-test" ]]; then
  log "infer-test  relative→abs→IK→dry execute"
  export PYTHONPATH="${SDK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  run python3 "${SCRIPT_DIR}/test_ee_infer_pipeline.py"
  exit 0
fi

ensure_lerobot_v30() {
  local ver
  ver="$(python3 -c "import json; print(json.load(open('${DST}/meta/info.json')).get('codebase_version',''))")"
  if [[ "${ver}" == v3* ]]; then
    log "lerobot   already ${ver}"
    return 0
  fi
  local parent name v30
  parent="$(dirname "${DST}")"
  name="$(basename "${DST}")"
  v30="${parent}/${name}_v30"
  if [[ -f "${v30}/meta/info.json" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open('${v30}/meta/info.json')).get('codebase_version','').startswith('v3') else 1)"; then
    log "lerobot   reuse ${v30}"
  else
    log "lerobot   v2.1 → v3.0  ${DST} → ${v30}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      DST="${v30}"
      return 0
    fi
    rm -rf "${v30}"
    mkdir -p "${v30}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "${DST}/" "${v30}/"
    else
      cp -a "${DST}/." "${v30}/"
    fi
    python3 - "${v30}/meta/info.json" <<'PY'
import json, math, sys
from pathlib import Path
path = Path(sys.argv[1])
info = json.loads(path.read_text())
episodes = int(info.get("total_episodes") or 0)
chunk = int(info.get("chunks_size") or 1000)
info["total_chunks"] = max(1, math.ceil(episodes / chunk)) if chunk else 1
n_video = sum(1 for feat in (info.get("features") or {}).values() if feat.get("dtype") == "video")
info["total_videos"] = episodes * n_video
path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")
print(f"total_chunks={info['total_chunks']} total_videos={info['total_videos']}")
PY
    activate_walloss
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    # 新 lerobot：--root 是父目录，实际读 Path(root)/repo_id
    # 旧 lerobot.scripts：--root 就是数据集目录本身
    if python -c "import lerobot.datasets.v30.convert_dataset_v21_to_v30" >/dev/null 2>&1; then
      python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
        --repo-id "${name}_v30" \
        --root "${parent}" \
        --push-to-hub=false
    elif python -c "import lerobot.scripts.convert_dataset_v21_to_v30" >/dev/null 2>&1; then
      python -m lerobot.scripts.convert_dataset_v21_to_v30 \
        --repo-id "${name}_v30" \
        --root "${v30}" \
        --push-to-hub=false
    else
      die "找不到 v21→v30 转换模块（lerobot.datasets.v30 或 lerobot.scripts）"
    fi
  fi
  DST="${v30}"
  # dtype=depth 没有对应视频时，新 lerobot 读 parquet 会失败。WALL-OSS 只用 RGB。
  python3 - "${DST}/meta/info.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if path.is_file():
    info = json.loads(path.read_text())
    feats = info.get("features") or {}
    dropped = [k for k, v in feats.items() if v.get("dtype") == "depth"]
    for key in dropped:
        del feats[key]
    if dropped:
        path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")
        print("dropped depth features", ",".join(dropped))
PY
}

if [[ "${WANT}" -ge 2 ]]; then
  ensure_lerobot_v30
  CFG_PATH="${DST}/sft.yml"
  NORM_PATH="${DST}/meta/walloss_norm_stats.json"
fi

if [[ "${WANT}" -lt 2 ]]; then
  log "停在 convert。EE 数据: ${DST}"
  exit 0
fi

# --- 2. yaml + norm ---
if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "would write ${CFG_PATH} and compute ${NORM_PATH}"
else
  [[ -f "${DST}/meta/info.json" ]] || die "EE 数据集不完整: ${DST}"
  log "write yaml ${CFG_PATH}"
  render_yaml >/dev/null
  activate_walloss
  [[ -d "${WALLX}" ]] || die "找不到 wall-x: ${WALLX}"
  [[ -f "${PRETRAINED}/model.safetensors" || -f "${RESUME}" ]] || die "找不到预训练权重: ${RESUME}"
  mkdir -p "$(dirname "${NORM_PATH}")" "${SAVE_PATH}"
  if [[ -f "${NORM_PATH}" && "${OVERWRITE}" -eq 0 ]]; then
    log "norm      skip（已有 ${NORM_PATH}）"
  else
    run python "${WALLX}/scripts/compute_norm_stats.py" --train_config "${CFG_PATH}"
  fi
  python3 - "${DST}" "${CFG_PATH}" "${NORM_PATH}" "${ACTION_TOKENIZER_PATH}" "${USE_FAST_TOKENIZER}" <<'PY'
import json, sys
from pathlib import Path
dst, cfg, norm, tok, use_fast = sys.argv[1:]
info = json.loads((Path(dst) / "meta" / "info.json").read_text())
marker = {
    "status": "PREPARED",
    "dataset": dst,
    "codebase_version": info.get("codebase_version"),
    "total_episodes": info.get("total_episodes"),
    "total_frames": info.get("total_frames"),
    "continuous": {
        "state": "observation.state.ee",
        "action": "action.ee",
        "norm_stats": norm,
    },
    "discrete": {
        "use_fast_tokenizer": use_fast == "true",
        "action_tokenizer_path": tok,
        "note": "同一段 action.ee 在 loader 里归一化后做 FAST，不另存一份 token 表",
    },
    "train_config": cfg,
}
(Path(dst) / "meta" / "PREPARED.json").write_text(
    json.dumps(marker, indent=2, ensure_ascii=False) + "\n"
)
print(dst)
PY
fi

if [[ "${WANT}" -lt 3 ]]; then
  log "停在 norm。下一步开训:"
  log "  bash $0 --src ${DST} --until train --nproc ${NPROC}"
  exit 0
fi

# --- 3. train ---
if [[ "${DRY_RUN}" -eq 0 ]]; then
  activate_walloss
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CHECKPOINT_OPTIMIZATION="${CHECKPOINT_OPTIMIZATION:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_ENTRY="${SETUP_DIR}/train/run_sft.py"
if [[ ! -f "${TRAIN_ENTRY}" ]]; then
  TRAIN_ENTRY="${WALLX}/wall_x/trainer/fsdp_trainer/train_fsdp.py"
fi

log "train     nproc=${NPROC} config=${CFG_PATH}"
# FSDP 入口要求 RANK/WORLD_SIZE；单卡也走 torchrun。
run python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_ENTRY}" \
  --config "${CFG_PATH}" \
  --log_to_file
