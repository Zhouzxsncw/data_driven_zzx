# 被 launch_pi05_*.sh source。机器路径放 host_env.sh（gitignore），YAML/py 与本机同一份。
if [[ -f "${BASH_SOURCE[0]%/*}/host_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${BASH_SOURCE[0]%/*}/host_env.sh"
fi
