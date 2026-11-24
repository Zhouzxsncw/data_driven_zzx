# 夹爪数据源说明

系统里有三类夹爪值：

| 名称 | 网络来源 | 含义 | 主要用途 |
|---|---|---|---|
| 外骨骼夹爪目标值 | 下位机 `exo_read_safety.py` → UDP `18131` → `robot_action[14:16]` | 操作者在外骨骼上捏出的左右目标开合比 | 控制真实夹爪；也可作为数据集/曲线夹爪字段 |
| 真实夹爪反馈值 | 下位机 `only_gripper.py` → UDP `30016` → `gripper_state[0:2]` | Dynamixel 真实左右夹爪实际位置 | 默认写入数据集和实时曲线 |
| 夹爪控制指令 | 上位机 `RobotArmBridge` → UDP `30015` → `only_gripper.py` | 发给真实夹爪电机的控制命令 | 实际驱动左右真实夹爪 |

## UDP 包布局（与下位机一致）

**30016 夹爪状态**（`only_gripper.py` 推送）：

```text
[left_ratio, right_ratio, timestamp, left_current_mA, right_current_mA]
```

数采解析后保留前两维 `[left, right]`（兼容旧 3 维包）。

**18131 robot_action**（`exo_read_safety.py` 推送）：

```text
[left7, right7, left_gripper, right_gripper, timestamp]   # 17 float64
```

数采解析后保留 16 维（去掉 timestamp）。

**30015 夹爪控制**（`RobotArmBridge` → `only_gripper.py`）：

```text
[flag, left_cmd, right_cmd]
```

## `gripper_source` 含义

```yaml
modalities:
  gripper_source: 0

  gripper_value_filter:
    enabled: true
    open_value: 1.0
    open_deadband: 0.08
    close_threshold: 0.85
    stable_samples: 2
    output_mode: binary
```

`gripper_source` 只控制数据集字段和实时曲线中的夹爪来源，不控制真实夹爪电机的控制来源。

| 配置 | `observation.state` / `action` 中启用的夹爪 | 实时夹爪曲线 | 是否需要 UDP `30016` |
|---:|---|---|---|
| `0` | 真实夹爪 `gripper_state[0]` / `[1]` | 真实夹爪反馈曲线 | 需要 |
| `1` | 外骨骼夹爪 `robot_action[14]` / `[15]` | 外骨骼夹爪目标曲线 | 不需要 |

## 真实夹爪如何被控制

真实夹爪控制链路始终是：

```text
外骨骼 robot_action[14] / [15]
        ↓
上位机 RobotArmBridge 左右独立防抖过滤
        ↓
UDP 30015: [0.0, left_gripper, right_gripper]
        ↓
下位机 only_gripper.py（dxl_id=1 左 / dxl_id=2 右）
        ↓
Dynamixel 真实夹爪运动
```

即使 `gripper_source: 0`（数据集存真实反馈），电机仍由外骨骼目标值控制。

## 推荐配置

```yaml
modalities:
  gripper_source: 0
  gripper_value_filter:
    enabled: true
    open_value: 1.0
    open_deadband: 0.08
    close_threshold: 0.85
    stable_samples: 2
    output_mode: binary
```

含义：

```text
observation.state 夹爪 = 真实夹爪实际位置（30016）
action 夹爪            = 同上（当前 gripper_source 同时作用于 state/action）
真实夹爪控制命令       = 外骨骼左右目标，过滤后经 30015 下发
```

双臂配置请同时打开 `state/action` 的 `left_gripper` 与 `right_gripper`。

## 夹爪语义与过滤

```text
1.0 = 全开 / 未按夹爪
0.0 = 闭合 / 按下夹爪
```

`modalities.gripper_value_filter` 在组帧前执行，同时影响 observation.state、action、实时曲线和最终写入值。

`robot_arm.gripper_filter` 只影响下发给真实夹爪的控制指令，与写入数据集的过滤相互独立。
