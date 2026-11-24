# τ0-WM 训练环境激活：source /home/xk/data_Driven/recipes/tau0_wm/env.sh
unalias -a 2>/dev/null || true
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
set +u
conda activate tau0wm
set +u

export TAU_ROOT="${TAU_ROOT:-/home/xk/VLA/tau-0-wm}"
export TAU_RECIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TAU_WEIGHTS="${TAU_WEIGHTS:-/data/pretrained/tau-0-wm}"
export WAN_ROOT="${WAN_ROOT:-/data/pretrained/Wan2.2-TI2V-5B}"
export TAU_RUNS="${TAU_RUNS:-/data/tau0_wm_runs}"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="12.0"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export PYTHONPATH="${TAU_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=offline

cd "${TAU_ROOT}"
echo "python=$(which python) $(python -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
python -c "import torch;print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')" 2>/dev/null
echo "TAU_ROOT=${TAU_ROOT}  weights=${TAU_WEIGHTS}  wan=${WAN_ROOT}  runs=${TAU_RUNS}"
