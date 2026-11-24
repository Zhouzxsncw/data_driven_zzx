# jiangsuan-4 宿主机 WALL-OSS（无 docker）
# source /nvme2n1/wall-oss/env.sh
set +u
source /usr/local/PPU_SDK/envsetup.sh
source /nvme2n1/wall-oss/venv/bin/activate
set +u
export ROOT=/nvme2n1/wall-oss
export WALLX=/nvme2n1/wall-oss/src/wall-x
export WALLX_CODE=/nvme2n1/wall-oss/src/wall-x
export WALLX_SKIP_CUDA_EXT=1
export HF_HOME="${HF_HOME:-/nvme2n1/wall-oss/hf}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PRETRAINED_PATH=/nvme2n1/wall-oss/pretrained/wall-oss-0.5
export PROCESSOR_PATH=/nvme2n1/wall-oss/pretrained/Qwen2.5-VL-3B-Instruct
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "${ROOT}"
echo "python=$(which python) ROOT=${ROOT}"
