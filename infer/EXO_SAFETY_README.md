# 外骨骼推理安全保障（exo_safety）

**分工**：除夹爪外，控制在上位机；下位机只做外骨骼读数 + 夹爪执行。

| 机器 | 职责 |
|------|------|
| **上位机** (192.168.1.12) | 推理、`direct_robot` API 控臂、Ctrl+1/2 脚踏板、下发 START/STOP、相机/视频 |
| **下位机** (192.168.1.13) | `lower_machine/`：夹爪 UDP 30015 + 外骨骼上行 18140（仅收到指令后） |

配置：`infer/config/pi05_left_arm_direct.yaml` → `exo_safety:`、`pedal:`、`direct_robot:`

---

## 1. 端到端流程

```
[下位机] python3 lower_machine/run.py
    ├─ only_gripper.py     → 30015 夹爪
    └─ exo_read_safety.py  → 18140（等 30026 START）
    └─ exo_read_safety.py  → 监听 30026，默认不上传

[上位机] infer/run_batch_infer.py（exo_safety.enabled: true）
    推理期：direct_robot API 控臂，不采用 exo

    Ctrl+1：
      ① 上位机 → 30026 START_STREAM
      ② 下位机开始 read() + 上行 18140
      ③ 暂停推理发臂，API 跟 exo
      ④ 终端询问「是否数采？」Y 写轨迹 / N 仅控制

    Ctrl+2：
      ① 上位机 → 30026 STOP_STREAM
      ② 下位机停止上传
      ③ movej 回 standard_init_joint_degrees，恢复推理
```

要点：

- **下位机无 Enter**，上传完全由上位机 Ctrl+1/Ctrl+2 驱动
- **接收 ≠ 正式采用**：只有 Ctrl+1 后 `run_takeover_step` 才用 exo 控臂
- **数采可选**：Ctrl+1 后 Y/N；N 不写 `exo_takeover_trajectory.jsonl`

---

## 2. 三种模式

| 模式 | 触发 | 下位机上传 | 上位机控臂 | 轨迹录制 |
|------|------|------------|------------|----------|
| `infer_only` | 会话启动 | ❌ | 推理 API | ❌ |
| `exo_takeover` | Ctrl+1 | ✅ | exo→API | Y 才录 |
| `exo_safety_infer` | Ctrl+2 | ❌ | 推理 API（先回初始位） | ❌ |

视频全程不断。

---

## 3. 启动

**下位机**（192.168.1.13）：

```bash
cd lower_machine
python3 run.py --gripper-only
python3 run.py
```

**上位机**：

```bash
conda activate openpi-pi05-infer
cd /home/cigai-24-3/infer
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 infer/run_batch_infer.py
```

yaml 中设置 `exo_safety.enabled: true`（默认 `false`，关闭时不影响现有流程）。

---

## 4. 配置示例

```yaml
direct_robot:
  enabled: true
  gripper_mode: udp

exo_safety:
  enabled: true
  exo_state_port: 18140
  exo_command_port: 30026

pedal:
  exo_takeover: ctrl+1
  exo_safety_infer: ctrl+2
```

---

## 5. 网络端口

| 端口 | 方向 | 用途 |
|------|------|------|
| **30026** | 上位机 → 下位机 | START(1) / STOP(0) |
| **18140** | 下位机 → 上位机 | 17 joint_rad + l_grip + r_grip + ts |
| **30015** | 上位机 → 下位机 | 夹爪 ratio |

协议常量：`infer/core/exo_protocol.py`

---

## 6. 产物路径

外骨骼相关文件写在**组目录**下（无 `test_XX` 子层），例如：

```
action-step_10/
├── exo_mode_markers.jsonl
└── exo_takeover/
    └── exo_takeover_trajectory.jsonl   # 仅 Ctrl+1 且选 Y
```

---

## 7. 代码索引

| 路径 | 机器 |
|------|------|
| [lower_machine/](lower_machine/README.md) | 下位机 |
| `infer/core/exo_safety.py` | 上位机协调 |
| `infer/core/exo_protocol.py` | 共享协议 |
| `infer/core/engine.py` | Ctrl+1 Y/N 数采 |

---

## 8. FAQ

**Q：怎么启动下位机？**  
A: `cd lower_machine && python3 run.py`（`--gripper-only` 仅夹爪）。
