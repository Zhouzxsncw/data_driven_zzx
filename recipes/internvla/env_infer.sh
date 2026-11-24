# source /home/xk/VLA/internvla_staging/env_infer.sh
# InternVLA-A1.5 服务端环境（conda internvla_a1_5）
unalias -a 2>/dev/null || true
# shellcheck disable=SC1091
source /home/xk/miniconda3/etc/profile.d/conda.sh
set +u
conda activate internvla_a1_5
set +u
# shellcheck disable=SC1091
source /data/internvla/env.sh

export INTERNVLA_CODE="${INTERNVLA_CODE:-/home/xk/VLA/InternVLA-A-series}"
export INTERNVLA_CKPT="${INTERNVLA_CKPT:-/data/internvla/outputs/00036_finetune_bs32/checkpoints/002000/pretrained_model}"
export VLM_MODEL_PATH="${VLM_MODEL_PATH:-/data/internvla/ckpts/Qwen3.5-2B}"
export CUDA_HOME="${CONDA_PREFIX:-/home/xk/miniconda3/envs/internvla_a1_5}"
export PATH="${CUDA_HOME}/bin:${PATH}"
# causal_conv1d_cuda 需要 conda 里的 libstdc++ (CXXABI_1.3.15)，不能用系统 /lib
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${INTERNVLA_CODE}/src:${INTERNVLA_CODE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

echo "python=$(which python) $(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
