# TongBot π₀.₅：第 7 关节持续旋转问题总结

TongBot 16D：`[左7 | 右7 | 左爪 | 右爪]`  
训练：`cigai-agilex-train` / `pi05_tongbot` / `tongbot_pi05_dual16`  
推理：`infer/config/lerobot/pi05_openpi_dual_arm_gripper_direct.yaml`  
复现：`.../041850/action-step_5`（修正 norm 后）；更早还有 `action-step_50`、错误 14D 配置 run

---

## 1. 结论

OpenPI π₀.₅ 双臂真机推理，在归一化修好之后，整体不再乱跑，但：

> **左右臂第 7 关节（腕）持续单向旋转。**

**根因：** 数采时腕关节标定偏置约 120°——下发做了 `+120°`，落盘的 `action` 仍是外骨骼逻辑角，`observation.state` 却是机械臂机身角，两者差约 120°。OpenPI 对关节做 **delta（`action − state`）**，把该固定偏差学成“每步再转 −120°”；推理 `cmd = state + delta` 形成追转。

**ACT / 老版左臂 LeRobot π₀.₅ 不转圈：** 学的是绝对角，不把 `action − state` 当监督。

**建议：**

1. **短期：** 推理锁 J7，或临时去掉读状态 −120°（保留下发 +120°）验证。  
2. **正解：** 统一数采 `action`/`state` 坐标系 → 重算 norm → **重训**。  
   - 已改：`data_collect/robot_arm_bridge.py` 读状态时对左右 J7 **−120°**（与推理对齐）。**旧数据仍污染，须重采或至少不能直接用旧数据训 delta。**

---

## 2. 现象时间线

| 阶段 | 现象 | 主因 |
|------|------|------|
| A | 右臂乱动、维度错乱 | 误用 Aloha **14D** 跑 TongBot **16D** ckpt |
| B | `(16,) (14,)` 广播报错 | 加载了 14D Aloha norm stats |
| C | 整体大幅乱动；夹爪常到 ±1 | 训练真实 quantile stats，推理用 **identity** |
| D | **整体不乱动，J7 转圈** | **数采 J7 坐标系不一致 × OpenPI delta** |

阶段 D（`action-step_5`，约 560 步）：

| 关节 | mean(指令 − 反馈) | 说明 |
|------|-------------------|------|
| L1–L6 / R1–R6 | ≈ 0.00～0.06 rad | 正常 |
| **L7** | **≈ −1.99 rad** | ~95% 贴近 **−120°** |
| **R7** | **≈ −2.06 rad** | ~95% 贴近 **−120°** |

累计转角约：L7 **−299°**，R7 **−347°**。

```text
当前 ≈ +0.16  → 指令 ≈ −1.94   （Δ ≈ −120°）
追到后再下一 chunk 再 −120° → 转圈
```

---

## 3. ±120° 与数采落盘

### 两套角

| 名称 | 含义 |
|------|------|
| 逻辑角 | 外骨骼 / 策略 16D `action` |
| 机身角 | RealMan 实际下发与读回 |

```text
下发：q_机身 = q_逻辑 + 120°   （仅左右 J7）
读回：q_逻辑 = q_机身 − 120°   （推理有；数采无）
```

16D：左 J7 = dim 6，右 J7 = dim 13。

### 数采实际写了什么（修复前）

```text
外骨骼 ──UDP──► 数据集 action（逻辑角，未写入 +120）
              └── bridge 下发时才 +120 ──► 机械臂

机械臂反馈 ──────────────────────────► 数据集 state（机身角，无 −120）
```

因此：`action[J7] ≈ state[J7] − 2.094 rad`。

**代码已确认与上图一致**（`modalities.py`：action←ROBOT_ACTION，state←ROBOT_STATE；`robot_arm_bridge` 仅下发 +120）。  
**已修复（新采数据）：** `_read_rm75_state` 对左右 J7 做 −120°，与推理 `direct_robot_io` 对齐。旧数据集仍为修复前分布。

### 推理 `direct_robot_io`

- 下发 `+120°`，读状态 `−120°`
- 与**新数采**落盘一致；**旧数据** state 无 −120，与推理观测空间在 J7 上仍不一致。

---

## 4. 为何只有 OpenPI delta 转圈

训练：`use_delta_joint_actions`，`delta_action_dims: [7,7,-1,-1]`。

```text
delta[J7] ≈ action − state ≈ −120°（近似常数）
推理：cmd ≈ state + (−120°) → 追转
```

| 方案 | 后端 / 配置 | 动作监督 | 对 J7 不一致 | 表现 |
|------|-------------|---------|--------------|------|
| OpenPI 双臂 | `pi05_backend=openpi`，delta `[7,7,-1,-1]` | **delta** | 极高 | **转圈** |
| 老左臂 π₀.₅ | `pi05_left_arm_direct.yaml`，默认 **lerobot** | **绝对角** | 低 | 不转 |
| ACT 双臂 | `act_aloha_...` | **绝对角** | 低 | 可正常 |
| OpenPI 左臂（若 delta `[7,-1]`） | `pi05_openpi_left_arm_direct.yaml` | delta | 高 | 左 J7 同样有风险 |

`keep_dims: [0..6,14]` 含左 J7，不是丢掉第 7 关节才不转。

---

## 5. 同次排查的其它问题

### 5.1 早期 14D Aloha 布局

| 项 | 正确 | 错误 run |
|----|------|----------|
| `openpi_action_dim` | 16 | 14 |
| `keep_dims` | `[0..15]` | Aloha 交错 14 维 |

### 5.2 norm_stats 未进 ckpt → 推理写 identity

`asset_id` 回落为绝对 `repo_id` 时：

```python
Path(ckpt/assets) / "/data1/..."  # 写飞到数据集目录，不进 ckpt
```

推理找不到匹配 16D stats 就合成 identity。维度匹配的真实 stats **不会**被覆盖；identity 一旦写入会当作合法 assets 继续用。

换上与 `train.log` 一致的真实 stats 后，整体乱动消失，收敛为 J7 转圈。

### 5.3 次要

夹爪数据质量、视觉 OOD 会影响策略，但不是 J7 转圈主因（转圈有明确 −120° 指纹）。

---

## 6. 配置与代码

| 用途 | 路径 |
|------|------|
| 出问题的双臂推理 | `infer/config/lerobot/pi05_openpi_dual_arm_gripper_direct.yaml` |
| 老左臂（不转） | `infer/config/lerobot/pi05_left_arm_direct.yaml` |
| ACT 双臂 | `infer/config/aloha/act_aloha_dual_arm_gripper_direct.yaml` |
| 数采下发 +120 | `data_collect/robot_arm_bridge.py` |
| 数采读状态无 −120 | 同上 `_read_rm75_state` |
| action/state 来源 | `data_collect/modalities.py` |
| 推理 ±120 | `infer/io/direct_robot_io.py` |
| norm 回退 | `infer/policies/openpi_pi05_runtime.py`（详解见 [`../policies/openpi_pi05_runtime.md`](../policies/openpi_pi05_runtime.md)） |

---

## 7. 建议

**P0 止血（不重训）**

1. 锁 J7：`desired = current`（dim 6/13）  
2. 或试验：保留下发 +120，暂时去掉读状态 −120（小范围安全测）  
3. 确认 `assets/norm_stats.json` 不是 identity  

**P1 正解**

- **A：** 数采读状态 J7 做 −120（与推理对齐）→ 重算 norm → 重训  
- **B：** action 改记机身角，推理去掉 ±120 → 重训  

旧 delta ckpt 的 J7 不可靠，需重训。

**P2 防再发**

1. `asset_id` 用短名，norm 写入 ckpt `assets/`  
2. 缺匹配 stats 时直接报错，禁止 identity  
3. 启动打印 delta_mask / action_dim / norm 来源  
4. 数采自检 `mean(action[J7]−state[J7])`  

---

## 8. 数值指纹

| 项 | 值 |
|----|-----|
| 120° | ≈ 2.094 rad |
| 数据 `action[J7]−state[J7]` | ≈ −2.094 |
| 训练 action R7 q01～q99（delta 后） | ≈ [−2.21, −1.98] |
| Identity norm | mean=0, q01=−1, q99=1 |
| 修 norm 后非 J7 \|Δ\| | ≈ 0.01～0.06 rad |
| 修 norm 后 J7 mean Δ | ≈ −2.0 rad |
