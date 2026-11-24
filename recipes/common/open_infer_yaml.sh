# 把推理客户端 YAML 弹到 Cursor。YAML 在进程启动时读入，改完再 launch。
#
#   source recipes/common/open_infer_yaml.sh
#   open_infer_yaml /abs/path.yaml
#
# SKIP_YAML_EDIT=1 / OPEN_INFER_YAML=0  不打开
open_infer_yaml() {
  local yaml="${1:-}"
  if [[ -z "$yaml" || ! -f "$yaml" ]]; then
    echo "[ERROR] open_infer_yaml: 不是文件: ${yaml:-<empty>}" >&2
    return 1
  fi
  if [[ "${SKIP_YAML_EDIT:-0}" == "1" || "${OPEN_INFER_YAML:-1}" == "0" ]]; then
    echo "[INFO] 跳过弹出 YAML: $yaml"
    return 0
  fi
  echo "[INFO] 推理参数 YAML: $yaml"
  if command -v cursor >/dev/null 2>&1; then
    cursor -r -g "$yaml" >/dev/null 2>&1 || cursor -r "$yaml" >/dev/null 2>&1 || cursor "$yaml" >/dev/null 2>&1 || true
  else
    echo "[WARN] 找不到 cursor。请在 IDE 打开: $yaml" >&2
  fi
  return 0
}
