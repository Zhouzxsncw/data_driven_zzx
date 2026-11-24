#!/usr/bin/env bash
# h100-103 上给 τ0-WM 开荒：venv(py3.11) + 22.9G 权重 + Wan2.2 三件复用 + 自检。
# 与 gpu-master 侧 recipes/tau0_wm/setup_env.sh 是同一套版本锁，差别只有承载机。
#
# 在 h100 上后台跑（push_to_h100.sh 会把整个 recipe 目录放到 local_scripts/ 下）：
#   setsid nohup bash /data1/tau0wm/local_scripts/h100/setup_h100.sh \
#       > /data1/tau0wm/setup_h100.out 2>&1 &
#
# 承载机事实（2026-09-15）：
#   - 无 conda；GWP /data1/gwp/envs/gigaworld-policy 是 py3.11 + torch 2.7.1+cu128
#     → 新建独立 venv，torch 从 GWP donor 拷，不改 GWP、也不走 pytorch.org
#   - hf-mirror 实测 ~18 MB/s（302 到 cas-bridge.xethub.hf.co），22.9G ≈ 20–25 min
#     本机代理 127.0.0.1:18897 当前拒绝连接，hf-mirror 直连即可
#   - /data1/dreamzero-pretrained/Wan2.2-TI2V-5B 原生布局齐全 → 软链，不重复下 32G
set -uo pipefail

ROOT="${TAU_H100_ROOT:-/data1/tau0wm}"
VENV="${ROOT}/envs/tau0wm"
REPO="${ROOT}/repo/tau-0-wm"
PRE="${ROOT}/pretrained"
WAN_SRC="${WAN_SRC:-/data1/dreamzero-pretrained/Wan2.2-TI2V-5B}"
DONOR_ENV="${DONOR_ENV:-/data1/gwp/envs/gigaworld-policy}"
DONOR_PY="${DONOR_PY:-${DONOR_ENV}/bin/python}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
LOG="${ROOT}/setup_h100.log"

mkdir -p "${ROOT}/envs" "${ROOT}/pretrained" "${ROOT}/repo" "${ROOT}/local_scripts" \
         "${ROOT}/datasets" "${ROOT}/outputs" "${ROOT}/logs"
: > "${LOG}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1          # hf-mirror 不代理 xet CAS，不禁会 401
export UV_LINK_MODE=copy

UV="${HOME}/.local/bin/uv"
[[ -x "${UV}" ]] || { log "没有 uv（${UV}）"; exit 1; }
[[ -x "${DONOR_PY}" ]] || { log "FATAL 缺 donor python: ${DONOR_PY}"; exit 2; }

copy_donor_torch() {
  local donor_site dest_site pyver pkg name
  pyver="$("${PY}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  donor_site="${DONOR_ENV}/lib/python${pyver}/site-packages"
  dest_site="${VENV}/lib/python${pyver}/site-packages"
  [[ -d "${donor_site}/torch" ]] || { log "FATAL donor 里没有 torch: ${donor_site}"; return 1; }
  # torch 运行时小依赖走清华源，体积可忽略
  "${UV}" pip install --python "${PY}" --default-index "${PIP_MIRROR}" \
    typing-extensions filelock fsspec jinja2 networkx sympy >>"${LOG}" 2>&1 \
    || log "WARN torch 小依赖安装异常"
  log "从 GWP donor 拷 torch 栈（${donor_site} → ${dest_site}）"
  # cusparselt 是顶层目录（libcusparseLt.so.0），不是 nvidia_* 前缀
  for pkg in torch torchvision torchaudio nvidia triton torchgen functorch cusparselt; do
    if [[ -e "${donor_site}/${pkg}" ]]; then
      log "  copy ${pkg}"
      rm -rf "${dest_site}/${pkg}"
      cp -a "${donor_site}/${pkg}" "${dest_site}/${pkg}"
    fi
    for meta in "${donor_site}"/${pkg}*.dist-info "${donor_site}"/${pkg}*.egg-info; do
      [[ -e "${meta}" ]] || continue
      rm -rf "${dest_site}/$(basename "${meta}")"
      cp -a "${meta}" "${dest_site}/$(basename "${meta}")"
    done
  done
  for dir in "${donor_site}"/nvidia_*; do
    [[ -e "${dir}" ]] || continue
    name="$(basename "${dir}")"
    rm -rf "${dest_site}/${name}"
    cp -a "${dir}" "${dest_site}/${name}"
  done
}

if [[ ! -x "${VENV}/bin/python" ]]; then
  log "[1/5] venv py3.11（donor=${DONOR_PY}）→ ${VENV}"
  "${DONOR_PY}" -m venv "${VENV}" >>"${LOG}" 2>&1 || { log "FATAL venv 创建失败"; exit 2; }
fi
PY="${VENV}/bin/python"
log "python = ${PY} $(${PY} -V 2>&1)"

if "${PY}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log "torch 已在 venv: $("${PY}" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
else
  copy_donor_torch || exit 2
  "${PY}" - <<'PY' 2>&1 | tee -a "${LOG}"
import torch
print(f"donor torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()} sm={torch.cuda.get_device_capability(0)}")
PY
  "${PY}" -c "import torch; assert torch.cuda.is_available()" \
    || { log "FATAL donor torch 拷完后 cuda 不可用"; exit 2; }
fi

log "[2/5] 先装 huggingface_hub，好让 22.9G 权重下载与其余 pip 并行"
"${UV}" pip install --python "${PY}" --default-index "${PIP_MIRROR}" \
    "huggingface_hub==0.35.3" fastparquet >>"${LOG}" 2>&1 || log "WARN huggingface_hub 安装异常"

log "[3/5] 后台 curl hf-mirror resolve 拉 sii-research/tau-0-wm → ${PRE}/tau-0-wm"
# snapshot_download / hf-xet 会跟到 us.aws.cdn.hf.co（H100 上 Connection refused）。
# curl 走 hf-mirror.com/.../resolve/main 会 302 到 cas-bridge.xethub.hf.co，实测 ~18 MB/s。
HF_DL_LOG="${ROOT}/logs/hf_download.log"
if [[ -f "${PRE}/tau-0-wm/simulator/diffusion_pytorch_model.bin" ]] \
   && [[ -f "${PRE}/tau-0-wm/diffusion_pytorch_model-00001-of-00002.bin" ]]; then
  log "权重已在，跳过下载"
  HF_PID=""
else
  mkdir -p "${PRE}/tau-0-wm/simulator" "$(dirname "${HF_DL_LOG}")"
  (
    set -euo pipefail
    HF_BASE="${HF_ENDPOINT}/sii-research/tau-0-wm/resolve/main"
    pull() {
      local rel="$1" dest="$2" min_bytes="$3"
      if [[ -f "${dest}" ]] && [[ "$(stat -c%s "${dest}")" -ge "${min_bytes}" ]]; then
        echo "[skip] ${rel} ($(stat -c%s "${dest}") bytes)"
        return 0
      fi
      echo "[get] ${rel}"
      curl -fL --retry 8 --retry-delay 3 --retry-all-errors \
        -C - --connect-timeout 20 \
        -o "${dest}" "${HF_BASE}/${rel}"
    }
    pull README.md "${PRE}/tau-0-wm/README.md" 100
    pull config.json "${PRE}/tau-0-wm/config.json" 100
    pull diffusion_pytorch_model.bin.index.json \
         "${PRE}/tau-0-wm/diffusion_pytorch_model.bin.index.json" 1000
    pull simulator/config.json "${PRE}/tau-0-wm/simulator/config.json" 100
    pull diffusion_pytorch_model-00002-of-00002.bin \
         "${PRE}/tau-0-wm/diffusion_pytorch_model-00002-of-00002.bin" 1000000000
    pull diffusion_pytorch_model-00001-of-00002.bin \
         "${PRE}/tau-0-wm/diffusion_pytorch_model-00001-of-00002.bin" 9000000000
    pull simulator/diffusion_pytorch_model.bin \
         "${PRE}/tau-0-wm/simulator/diffusion_pytorch_model.bin" 11000000000
    echo DONE
  ) > "${HF_DL_LOG}" 2>&1 &
  HF_PID=$!
  log "hf download pid=${HF_PID}（日志 ${HF_DL_LOG}；curl 22.9G @~18MB/s ≈ 20–25 min）"
fi

log "[4/5] 训练主依赖（清华源；torch 已从 donor 就位，不再拉 cu128 轮子）"
"${UV}" pip install --python "${PY}" --default-index "${PIP_MIRROR}" \
    "accelerate==1.12.0" "diffusers==0.35.2" "transformers==4.57.6" "tokenizers==0.22.2" \
    "huggingface-hub==0.35.3" "safetensors==0.6.2" "datasets==4.1.1" "pyarrow==23.0.1" \
    "pandas==2.3.1" "numpy==2.2.6" "einops==0.8.1" "omegaconf==2.3.0" "PyYAML==6.0.2" \
    "pyyaml-include==1.4.1" "av==15.1.0" "kornia==0.8.2" "tensorboard==2.20.0" \
    "tqdm==4.67.1" "scipy==1.15.3" "matplotlib==3.10.5" "moviepy==1.0.3" \
    "imageio==2.37.2" "imageio-ffmpeg==0.6.0" "msgpack==1.1.2" "websockets==14.1" \
    "sentencepiece==0.2.0" "ftfy==6.3.1" "pillow==11.3.0" \
    "opencv-python-headless==4.10.0.84" "filelock==3.20.1" "requests==2.32.5" \
    "psutil==7.1.3" packaging termcolor jsonlines deepdiff \
    "draccus==0.10.0" dm-tree gymnasium >>"${LOG}" 2>&1 \
  || log "WARN：主依赖有冲突，见日志"

# lerobot 必须 --no-deps，否则会把 torch 回退成 cu12x 默认轮子
"${UV}" pip install --python "${PY}" --default-index "${PIP_MIRROR}" --no-deps \
    "lerobot==0.4.4" >>"${LOG}" 2>&1 || log "WARN lerobot 安装异常"

log "[5/5] Wan2.2 三件软链 + 等权重 + 自检"
if [[ -f "${WAN_SRC}/models_t5_umt5-xxl-enc-bf16.pth" && -f "${WAN_SRC}/Wan2.2_VAE.pth" ]]; then
  mkdir -p "${PRE}/Wan2.2-TI2V-5B/google"
  ln -sfn "${WAN_SRC}/models_t5_umt5-xxl-enc-bf16.pth" "${PRE}/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
  ln -sfn "${WAN_SRC}/Wan2.2_VAE.pth"                  "${PRE}/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  ln -sfn "${WAN_SRC}/google/umt5-xxl"                 "${PRE}/Wan2.2-TI2V-5B/google/umt5-xxl"
  log "Wan2.2 三件 → ${PRE}/Wan2.2-TI2V-5B（软链 ${WAN_SRC}）"
else
  log "WARN：${WAN_SRC} 里没有 T5/VAE，需要另找或补下（infer 与训练都要用）"
fi

if [[ -n "${HF_PID:-}" ]]; then
  log "等待权重下载结束（pid ${HF_PID}）..."
  wait "${HF_PID}" && log "权重下载完成" || log "WARN hf download 退出码非 0，见 ${HF_DL_LOG}"
fi
du -sh "${PRE}/tau-0-wm" 2>/dev/null | tee -a "${LOG}"

cd "${REPO}" 2>/dev/null || log "WARN：${REPO} 还没同步（由 gpu-master 侧 rsync）"
# fork 补丁（上游 forward_pass 死导入，不打补丁 runner.posttrain 连 import 都过不去）
if [[ -f "${REPO}/main.py" ]]; then
  "${PY}" "${ROOT}/local_scripts/patch_tau_repo.py" --repo "${REPO}" 2>&1 | tee -a "${LOG}"
else
  log "WARN：repo 未就位，补丁留到 relay_watch.sh 的 preflight 再打"
fi
"${PY}" - <<'PYEOF' 2>&1 | tee -a "${LOG}"
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available(), "| sm", torch.cuda.get_device_capability(0))
import lerobot, transformers, diffusers, accelerate, av, kornia, einops, omegaconf
print("lerobot", lerobot.__version__, "| transformers", transformers.__version__,
      "| diffusers", diffusers.__version__, "| accelerate", accelerate.__version__, "| av", av.__version__)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
for m in ("_ensure_hf_dataset_loaded", "_get_query_indices", "_query_hf_dataset",
          "_get_query_timestamps", "_query_videos"):
    assert hasattr(LeRobotDataset, m), m
print("lerobot 私有 API 齐备")
try:
    from runner.posttrain import Trainer
    print("runner.posttrain import OK")
except Exception as e:
    print("runner.posttrain 还不可用（repo 未同步或缺依赖）：", type(e).__name__, e)
PYEOF

log "SETUP_DONE"
