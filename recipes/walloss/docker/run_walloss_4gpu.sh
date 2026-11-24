#!/bin/bash
# jiangsuan-4 是 K8s Pod（containerd），没有 docker。
# WALL-OSS 装在宿主机 /nvme2n1/wall-oss，与 16 卡容器布局相同。
echo "[ERR] jiangsuan-4 has no docker. Use:"
echo "  bash recipes/walloss/bootstrap_jiangsuan4.sh"
exit 1
