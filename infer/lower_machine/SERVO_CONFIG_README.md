# 舵机更换配置指南

更换夹爪/外骨骼舵机后，用本脚本完成波特率与 ID 配置。

```bash
python3 lower_machine/configure_servo.py <子命令> [参数]
```

## 前置条件

| 项 | 说明 |
|---|---|
| 下位机 | `lab@192.168.1.13`，SSH 密码通过环境变量 `EXO_SSH_PASSWORD` 传入 |
| Python 环境 | 下位机 `dynamixel_sdk` 装在 conda env `tong`，脚本会自动使用 |
| 串口空闲 | **配置前必须停掉运行中的控制程序**（`run.py` / `only_gripper.py` 等），否则报 `multiple access on port` |

## 换舵机完整流程

假设换右夹爪舵机（目标：ID=2，4M）：

1. **停掉下位机控制程序**（占用串口的进程）
   ```bash
   # 下位机上
   ps aux | grep -E 'run.py|only_gripper|exo_read'
   kill <PID>        # 或停掉启动 run.py 的终端 Ctrl+C
   ```

2. **把新舵机接到 ttyUSB2 总线**（U2D2 菊花链）

3. **扫描确认总线现状**
   ```bash
   python3 lower_machine/configure_servo.py scan
   ```

4. **一键配置**（统一波特率 + 改 ID + 自动验证）
   ```bash
   python3 lower_machine/configure_servo.py auto --baud 4000000 --id 2
   ```
   如果总线是 4M 已有舵机（如左爪），`auto` 只改新舵机的波特率；`--id` 仅在
   总线上**唯一一个** ID 不符的舵机时生效（目标 ID 已被占用会提示改用 `swap`）。

5. **确认最终状态**
   ```bash
   python3 lower_machine/configure_servo.py scan
   ```
   期望：`ID=1` 和 `ID=2` 各一个，都在 4M 下响应。

## 子命令参考

```bash
# 全波特率扫描（9600 ~ 4M），列出所有舵机的 ID/型号/波特率
configure_servo.py scan

# 一键配置：统一总线波特率为 4000000，并把唯一的新舵机 ID 改为 2
configure_servo.py auto --baud 4000000 --id 2

# 只统一波特率，不动 ID（多舵机总线）
configure_servo.py auto --baud 4000000

# 互换两个舵机 ID（左右装反时用；自动用临时 ID 过渡，全程无冲突）
configure_servo.py swap --id-a 1 --id-b 2

# 单步：改波特率（--connect 指定当前能通信的波特率，默认 57600）
configure_servo.py setbaud --id 1 --baud 4000000 --connect 57600

# 单步：改 ID（--connect 默认 4000000）
configure_servo.py setid --id 1 --new-id 2 --connect 4000000

# 检查某个舵机是否响应
configure_servo.py ping --id 1 --baud 4000000
```

## 硬件/协议要点（踩坑总结）

### 总线拓扑

| 串口 | 波特率 | 设备 |
|------|--------|------|
| `/dev/ttyUSB1` | 1M | 外骨骼 19 舵机（ID 0-16, 20, 21，型号 1200/XC330-M181） |
| `/dev/ttyUSB2` | 4M | 夹爪左右各一（ID 1=左爪, 2=右爪，型号 1020/XM430-W350 兼容款） |

### 国产舵机波特率表（重要！）

这批国产舵机（model 1020 兼容款）的波特率寄存器表**与 Dynamixel SDK 标准不同**：

| 目标波特率 | 写入寄存器值（EEPROM addr 8） |
|-----------|------------------------------|
| 57600 | **1** |
| 4000000 | **6** |

- 按 SDK 标准写 4M=9 会被固件拒绝（错误码 `de=4`），**4M 必须写 6**
- 新舵机出厂默认：波特率 57600（值 1）、ID=1
- 舵机 ID 地址：EEPROM addr 7；波特率地址：addr 8（协议 2.0）
- 左爪原装舵机出厂即 4M（值 6）；换右爪时把新舵机从 57600（值 1）改为 4M（值 6）即可

### 注意事项

- **改 ID 前先想好目标 ID 是否被占用**：两个舵机 ID 相同会互相冲突（扫描只显示其中一个）。互换用 `swap` 命令，不要手动两个都改
- 写入 EEPROM 前请确保舵机未堵转、力矩未开启（程序停着就是安全的）
- 配置期间不要让下位机控制程序（`run.py`）运行，会占用串口
- 更换外骨骼舵机（ttyUSB1 @ 1M）同理：`scan` 后 `setbaud --connect 57600` 目标 1M 的映射需确认（本表仅实测 57600/4M 两档）

## 常见问题

| 现象 | 原因/处理 |
|------|----------|
| `multiple access on port` | 有程序占着串口，停掉 `run.py` / `only_gripper.py` 再试 |
| 写波特率报 `de=4` | 值不在国产舵机表内（如 9），确认用 6（4M）/ 1（57600） |
| 扫描只看到 1 个舵机 | 两个舵机 ID 冲突（都是 1）或波特率不一致，先 `scan` 看每档波特率下的 ID |
| 夹爪左右动作反了 | ID 和物理位置不匹配，`swap --id-a 1 --id-b 2` 互换 |
| 目标 ID 已被占用 | 用 `swap` 互换，或先把占用者改到临时 ID 再配置 |
