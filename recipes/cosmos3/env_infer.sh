# source /home/xk/VLA/cosmos3_staging/env_infer.sh
export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
export COSMOS_DIR="/home/xk/VLA/cosmos"
export COSMOS3_MODEL="/home/xk/VLA/cosmos_h100_setup/Cosmos3-Nano"
export COSMOS3_CKPT="${COSMOS3_CKPT:-/home/xk/VLA/h100_ckpts/cosmos3_00073/iter_000001500}"
export WAN_VAE_PATH="/home/xk/VLA/dreamzero_staging/pretrained/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
export COSMOS3_DIFFUSERS="/home/xk/VLA/cosmos_h100_setup/diffusers"
export COSMOS3_FRAMEWORK="/home/xk/VLA/cosmos-framework"
export PYTHONPATH="/home/xk/VLA/cosmos-framework:${PYTHONPATH:-}"
source "/data/cosmos3-infer/bin/activate"
echo "python=$(which python) $(python -c 'import sys; print(sys.version.split()[0])')"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
