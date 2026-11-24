#!/bin/bash
# H100 上部署 ACoT-VLA（离线迁移）
# 用法: bash deploy_acot_h100.sh
set -euo pipefail

BASE=/data1/ACoT-VLA
PYROOT=$BASE/uv_python/cpython-3.11.14-linux-x86_64-gnu

cd $BASE

echo "=== 1. 解压代码 ==="
if [ ! -d $BASE/src ]; then
  tar xzf acot_vla_code.tar.gz --strip-components=1 -C $BASE
fi
ls src/ | head -3

echo "=== 2. 解压 uv python 3.11 ==="
if [ ! -d $PYROOT ]; then
  mkdir -p $BASE/uv_python
  tar xf uv_python311.tar -C $BASE/uv_python
fi
$PYROOT/bin/python3.11 --version

echo "=== 3. 解压 venv ==="
if [ ! -d $BASE/.venv/lib ]; then
  tar xf acot_venv.tar -C $BASE
fi

echo "=== 4. 修复 venv 路径 ==="
# 4.1 python 符号链接 → 本地 uv python
ln -sf $PYROOT/bin/python3.11 $BASE/.venv/bin/python
ln -sf python $BASE/.venv/bin/python3
# 4.2 pyvenv.cfg
cat > $BASE/.venv/pyvenv.cfg << EOF
home = $PYROOT/bin
implementation = CPython
uv = 0.12.6
version_info = 3.11
include-system-site-packages = false
prompt = openpi
EOF
# 4.3 修复所有脚本的 shebang（/home/xk/ACoT-VLA → /data1/ACoT-VLA）
grep -rl '^#!/home/xk/ACoT-VLA' $BASE/.venv/bin/ 2>/dev/null | while read f; do
  sed -i "s|^#!/home/xk/ACoT-VLA|#!$BASE|" "$f"
done
echo "venv 路径修复完成"

echo "=== 5. 解压数据集 ==="
if [ ! -d $BASE/dataset/00036_20260804_RM_01_yj_v21 ]; then
  tar xzf dataset_00036.tar.gz -C $BASE/dataset/
fi
ls $BASE/dataset/

echo "=== 6. 解压 pi05_base 权重 ==="
if [ ! -d $BASE/pretrained_weight/pi05_base/params ]; then
  tar xzf pi05_base_params.tar.gz -C $BASE/pretrained_weight/
fi
du -sh $BASE/pretrained_weight/pi05_base/params

echo "=== 7. 更新 config.py 路径（H100 版） ==="
CFG=$BASE/src/openpi/training/config.py
# 数据集路径
sed -i 's|/home/xk/dataset/00036_20260804_RM_01_yj_v21|/data1/ACoT-VLA/dataset/00036_20260804_RM_01_yj_v21|g' $CFG
# pi05_base 权重路径
sed -i 's|/home/xk/agx/PI0.5/pretrained_weight/openpi-assets/checkpoints/pi05_base/params|/data1/ACoT-VLA/pretrained_weight/pi05_base/params|g' $CFG
grep -n "data1/ACoT-VLA" $CFG | head -3

echo "=== 8. 验证 jax GPU ==="
$BASE/.venv/bin/python -c "
import jax
print('jax', jax.__version__)
print('devices:', jax.devices())
" 2>&1 | tail -3

echo "=== 9. action-dim loss probe ==="
# 本机 data_Driven 同步到 H100 后执行；已焊进 fork 则 --check 直接 OK
if [ -f /home/xk/data_Driven/scripts/patch_acot_loss_probe.py ]; then
  python3 /home/xk/data_Driven/scripts/patch_acot_loss_probe.py $BASE
  python3 /home/xk/data_Driven/scripts/patch_acot_loss_probe.py $BASE --check
elif [ -f "$BASE/../data_Driven/scripts/patch_acot_loss_probe.py" ]; then
  python3 "$BASE/../data_Driven/scripts/patch_acot_loss_probe.py" $BASE --check
else
  echo "SKIP: patch_acot_loss_probe.py 不在本机；确认 acot_vla.py 已含 _acot_action_debug"
fi

echo ""
echo "=== 部署完成 ==="
echo "Python: $BASE/.venv/bin/python"
echo "训练启动: cd $BASE && nohup .venv/bin/python scripts/train.py config.acot_tongbot_00036 > train_h100.log 2>&1 &"
