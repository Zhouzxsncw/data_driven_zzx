# infer-tk: PI0.5 server/client environment
_PI05_RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export INFER_ROOT="$_PI05_RECIPE_ROOT/infer"
export PI05_PYTHON=/home/cigai22/anaconda3/envs/openpi-pi05-infer/bin/python
export OPENPI_INFER_PYTHON=/home/cigai22/anaconda3/envs/openpi-pi05-infer/bin/python
export PI05_LIBERO_PYTHON=/home/cigai22/anaconda3/envs/openpi-pi05-infer/bin/python
export OPENPI_DISABLE_TORCH_COMPILE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="$_PI05_RECIPE_ROOT/recipes/pi05/tools:$PATH"
unset _PI05_RECIPE_ROOT
