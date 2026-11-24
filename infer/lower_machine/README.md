# 下位机（192.168.1.13）

空机全套硬件调通（臂/相机/夹爪/外骨骼 → 模型）：仓库根 [docs/HARDWARE_BRINGUP.md](../../docs/HARDWARE_BRINGUP.md)。

## 启动

```bash
cd ~/TongGenesis/scripts/lower_machine
python3 run.py --gripper-only   # 仅夹爪（推理常用）
python3 run.py                  # 夹爪 + 外骨骼
```

**整目录同步**，至少包含：

```
lower_machine/
├── _bootstrap.py
├── run.py
├── only_gripper.py
├── gripper_udp_packet.py
├── gripper_motor_read.py
├── exo_kinematics.py
├── exo_read_safety.py
└── TG_exo/
    ├── __init__.py
    └── tg_exo_2.py
```

勿用 TongGenesis 根目录下的旧 `TG_exo/`。

## 逻辑说明

| 进程 | 作用 |
|------|------|
| `only_gripper.py` | 监听 30015，控左右真机夹爪；**推送 30016** 直读 Dynamixel id=1/2 |
| `exo_read_safety.py` | 读 USB 外骨骼；等 30026：`1`正式接管 / `2`人工对齐 / `3`自动对齐驱动；上行 18140；正式接管时另推 18131+本机 30015；自动对齐收 **30027** 目标角 |

自动对齐时外骨骼关节会主动转动：**请勿穿戴或用力对抗**。若舵机写角失败，日志会列出 `MultiDynamixel` 可用方法名。

**数采夹爪约定**（与 infer 外骨骼接管一致）：
- `observation.state` 夹爪 → 30016（真机 Dynamixel）
- `action` 夹爪 → 18140 / 18131（外骨骼）

exo 进程异常退出时，**夹爪继续运行**（推理不受影响）。

## 默认 IP / 端口

|  |  |
|--|--|
| 下位机 | 192.168.1.13 |
| 上位机 | 192.168.1.12 |
| 夹爪控制 | 30015 |
| 夹爪状态回传 | 30016 |
| robot_state | （数采/推理均不上位机 UDP；关节由本机 RealMan SDK 读取） |
| robot_action | 18131 |
| exo 上行 | 18140 |
| exo 指令 | 30026 |
