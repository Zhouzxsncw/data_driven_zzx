#!/usr/bin/env bash
# 一次 sudo：udev + dialout，并探测本机 IP / 睿尔曼 / RealSense，写进 local yaml
#   bash recipes/lingbot_va/install_robot_camera_sdk.sh
set -eu
unalias -a 2>/dev/null || true
PATH="/usr/sbin:/usr/bin:/sbin:/bin:/home/xk/miniconda3/envs/lingbotva/bin:${PATH:-}"

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo --preserve-env=LINGBOTVA_PYTHON,OPENPI_INFER_PYTHON,LINGBOT_VA_LOCAL_YAML /bin/bash "$0" "$@"
fi

UDEV_SRC="/home/xk/VLA/lingbot-va-infer/sdk_udev"
PYTHON_VA="${LINGBOTVA_PYTHON:-/home/xk/miniconda3/envs/lingbotva/bin/python}"
REAL_USER="${SUDO_USER:-xk}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
export HOME="$REAL_HOME"

echo "=== 1) RealMan RM_API ==="
if [[ ! -f "$REAL_HOME/RM_API/Robotic_Arm/rm_robot_interface.py" ]]; then
  echo "缺少 $REAL_HOME/RM_API" >&2
  exit 1
fi
echo "  OK $REAL_HOME/RM_API"

echo "=== 2) udev + dialout ==="
install -d /etc/udev/rules.d
cp -a "$UDEV_SRC/99-realsense-libusb.rules" /etc/udev/rules.d/
cp -a "$UDEV_SRC/99-realsense-d4xx-mipi-dfu.rules" /etc/udev/rules.d/
cat >/etc/udev/rules.d/99-dynamixel-usb.rules <<'EOF'
# Robotis U2D2 / FTDI 夹爪
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", MODE="0666", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="fff1", MODE="0666", GROUP="dialout"
EOF
udevadm control --reload-rules
udevadm trigger
if ! id -nG "$REAL_USER" | grep -qw dialout; then
  usermod -aG dialout "$REAL_USER"
  echo "  已把 $REAL_USER 加入 dialout，重新登录后夹爪 /dev/ttyUSB* 才免 sudo"
else
  echo "  $REAL_USER 已在 dialout 组"
fi
ls -l /etc/udev/rules.d/99-realsense-libusb.rules \
      /etc/udev/rules.d/99-realsense-d4xx-mipi-dfu.rules \
      /etc/udev/rules.d/99-dynamixel-usb.rules

echo "=== 3) SDK import ==="
runuser -u "$REAL_USER" -- "$PYTHON_VA" -c 'import sys; from pathlib import Path; sys.path.insert(0, str(Path.home()/"RM_API")); import pyrealsense2 as rs, dynamixel_sdk, serial; from Robotic_Arm.rm_robot_interface import rm_thread_mode_e; print("pyrealsense2", getattr(rs, "__version__", "ok")); print("dynamixel_sdk OK"); print("pyserial", serial.__version__); print("RM_API OK")'

echo "=== 4) 探测 IP / 机械臂 / 相机并写入 yaml ==="
runuser -u "$REAL_USER" -- "$PYTHON_VA" /home/xk/VLA/lingbot-va-infer/pair_robot_site.py

echo "done"
