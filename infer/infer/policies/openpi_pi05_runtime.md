# OpenPI π₀.₅ Runtime 开发者文档

对应代码：[`openpi_pi05_runtime.py`](openpi_pi05_runtime.py)  
调用方：[`pi05_adapter.py`](pi05_adapter.py)（`pi05_backend: openpi`）  
Vendored 源码：[`openpi/`](openpi/)  
环境安装：[OPENPI_ENV.md](../docs/OPENPI_ENV.md)  
故障复盘：[REPORT_PI05_J7_SPIN_AND_INFER_ISSUES.md](../docs/REPORT_PI05_J7_SPIN_AND_INFER_ISSUES.md)

---

## 1. 定位与设计目标

`OpenpiPi05Runtime` 是本仓 **进程内** 跑 OpenPI π₀.₅ 的胶水层，职责对标 ACT 的 `aloha_*_runtime`：

| 职责 | 说明 |
|------|------|
| 路径注入 | 把 `infer/policies` 放进 `sys.path`，使 `import openpi.*` 命中 vendored 包 |
| 懒加载 | **仅** `load()` 时 import `openpi` / `jax`；LeRobot / ACT / DP 路径不拉 JAX |
| 配置覆盖 | YAML 可覆盖训练 config 的 `adapt_to_pi`、`delta_action_dims`（frozen dataclass → `dataclasses.replace`） |
| Norm 解析 | 无网也可启动；按维度过滤 14D/16D；必要时合成 identity |
| Obs 组装 | 相机短名映射、resize-with-pad、CHW uint8、`{state, images, prompt}` |
| Infer | `policy.infer(obs)` → `[T, D]`，截断/校验与 `openpi_action_dim` 对齐 |

**不做：** 真机 UDP、15D↔16D expand、夹爪通道、RTC、数采。这些在 `Pi05Adapter` / `InferEngine` / `direct_robot_io`。

---

## 2. 端到端调用链

```text
YAML: policy.type=pi05, pi05_backend=openpi
  → InferSession / batch_infer
    → Pi05Adapter.load()
      → Pi05Adapter._load_openpi()
        → OpenpiPi05Runtime(...)
        → runtime.load()
             · resolve_openpi_src_path → sys.path
             · get_config(openpi_config_name)
             · override adapt_to_pi / delta_action_dims
             · _resolve_norm_stats_for_policy(...)
             · create_trained_policy(..., norm_stats=...)
    → 单帧:
         build_frame(images, state[keep_dims], prompt)
           → runtime.build_observation(...)
           → runtime.infer(obs) → actions [T, D]
         expand_action_to_15d → direct_robot_io 下发
```

关键约定：

- 模型看到的 `state` / `actions` 维数 = `openpi_action_dim`，且 **必须等于** `len(keep_dims)`。
- TongBot 双臂默认 **16D**：`[左7 | 右7 | 左爪 | 右爪]`，索引 `0..15`。
- 真机控制层另有 15D/17D 布局与 J7 ±120° 偏置（见 §8），与模型维是两层事。

---

## 3. 模块结构

### 3.1 公开 API

| 符号 | 类型 | 用途 |
|------|------|------|
| `resolve_openpi_src_path(repo_path=None)` | 函数 | 返回应插入 `sys.path` 的目录（内含 `openpi/`） |
| `openpi_image_key(camera_name)` | 函数 | AgileX/本仓短名 → OpenPI 观测键 |
| `OpenpiPi05Runtime` | 类 | 加载 + 建 obs + infer |

### 3.2 内部工具（测试会直接测）

| 符号 | 用途 |
|------|------|
| `_to_hwc_uint8` | CHW/HWC、float/uint8 → HWC uint8 |
| `_resize_with_pad_numpy` | PIL resize-with-pad（**不** import openpi/jax） |
| `_apply_adapt_to_pi_override` | 覆盖 `TrainConfig.data.adapt_to_pi` |
| `_apply_delta_action_dims_override` | 覆盖 `TrainConfig.data.delta_action_dims` |
| `_try_load_norm_stats` | 本地或已缓存 gs 路径加载；失败返回 `None` |
| `_norm_stats_match_action_dim` | 校验 state/actions 末维 == `action_dim` |
| `_accept_norm_stats` | 维不匹配则打 warning 并丢弃 |
| `_make_identity_norm_stats` | 合成 quantile 恒等统计 |
| `_materialize_norm_stats` | 写回 `ckpt/assets/.../norm_stats.json` |
| `_resolve_norm_stats_for_policy` | 四级回退解析 |

### 3.3 相机名映射

```text
left / cam_left     → left_wrist_0_rgb
front / base / cam_mid → base_0_rgb
right / cam_right   → right_wrist_0_rgb
```

YAML `openpi_camera_names` 顺序必须与 `Pi05Adapter` 传入的 `images[]` 顺序一致。Adapter 侧再把短名映到 `observation.cam_*`（录像/同步用），与 OpenPI 键是两套命名。

---

## 4. `OpenpiPi05Runtime` 生命周期

### 4.1 构造参数

```python
OpenpiPi05Runtime(
    ckpt_dir=...,              # 含 params/ 或 model.safetensors
    config_name=...,           # 如 pi05_aloha；须能 get_config
    repo_path=None,            # 默认 vendored；测试可指向临时树
    camera_names=[...],
    action_dim=16,             # = openpi_action_dim
    action_horizon=50,         # YAML 初值；load 后以 TrainConfig.model 为准
    image_size=224,
    default_prompt="null",
    adapt_to_pi=False,         # TongBot 必须 False
    delta_action_dims=[7,7,-1,-1],  # 与训练一致；None=不覆盖
)
```

### 4.2 `load()` 逐步行为

1. **`_ensure_openpi_on_path`**：插入 vendored 路径。
2. **Late import**：`openpi.policies.policy_config`、`openpi.training.config`。
3. **`get_config(config_name)`** → 应用 `adapt_to_pi` / `delta_action_dims` 覆盖。
4. **安全检查**：`adapt_to_pi=True` 且 `action_dim != 14` → 直接 `ValueError`（AlohaInputs 会索引 `state[[6,13]]`）。
5. **horizon**：若 `TrainConfig.model.action_horizon` 与 YAML 不一致，**以模型为准**并 warning。
6. **打日志**：`config / ckpt / action_dim / horizon / adapt_to_pi / delta_action_dims`（排查必备）。
7. **`_resolve_norm_stats_for_policy`**（见 §5）。
8. **`create_trained_policy(train_cfg, ckpt_dir, default_prompt=..., norm_stats=...)`**。

**00108 FT/DEPTH ckpt（2026-09-23 这批）：** 训练进程开训时模态标签是旧文案 `FT LEFT: ` / `FT RIGHT: `（以及 `DEPTH HEAD:` / `DEPTH WRIST LEFT:`）。盘上 YAML 后来改过 `FORCE TORQUE LEFT/RIGHT`，**推理不要跟着改**，否则和这批权重 prefix 对不齐。`build_observation` 也还没接 DEPTH/FT，接的时候标签跟 ckpt 当时的 tokenizer 走，不要跟最新训练源码。

### 4.3 `build_observation(images, state, prompt)`

输入约定：

- `images`：长度 ≥ `len(camera_names)`；HWC 或 CHW，uint8 或 float。
- `state`：一维，长度 == `action_dim`（已是 `keep_dims` 切片后的模型维）。
- `prompt`：空则回落 `default_prompt`。

处理：

```text
每路图 → HWC uint8 → resize_with_pad(H,W) → CHW uint8
→ images[openpi_key] = chw
→ return { "state": float32[D], "images": {...}, "prompt": str }
```

与 `cigai-agilex-train` 的 joint-ros2 推理一致；差异仅在 **state 维/顺序**（TongBot 16D vs Aloha 14D）。

### 4.4 `infer(obs)`

```text
policy.infer(obs)["actions"] → float32
  · 1D → reshape (1, D)
  · D < action_dim → ValueError
  · D > action_dim → 截断到前 action_dim 列
  · 用实际 T 更新 _action_horizon
→ return (raw_copy, actions)  # 目前二者同内容
```

Delta→绝对角转换在 OpenPI **policy 输出 transform**（`AbsoluteActions`）内完成，runtime 拿到的已是绝对关节指令（在模型坐标系下）。

---

## 5. Norm stats 解析（高频踩坑）

### 5.1 优先级

```text
1. ckpt_dir/assets/<asset_id>/norm_stats.json   （asset_id 为 "" 或 "." 时直接在 assets/）
2. TrainConfig.data.assets.assets_dir + asset_id （本地存在才用；gs 仅在 maybe_download 成功时）
3. 默认 gs://openpi-assets/.../pi05_base + trossen（已缓存/可达且维匹配才用）
4. 合成 identity（q01=-1, q99=1 → quantile 映射近似恒等），并尝试写回 ckpt/assets
```

任意候选若 **state/actions 末维 ≠ `openpi_action_dim`**，整份丢弃并 warning。典型场景：TongBot 16D ckpt 误加载 Aloha/trossen **14D** stats → 曾出现 `(16,) (14,)` 广播错误。

### 5.2 Identity 的含义与危险

π₀.₅ 常用 quantile 归一化。Identity：

```text
mean=0, std=1, q01=-1, q99=1
→ (x - q01) / (q99 - q01) * 2 - 1 ≈ x
```

适用：训练时 assets 为空 / 刻意 skip norm。  
**不适用：** 训练用了真实数据集 quantile，推理却 identity → 整体动作量级错乱（夹爪易飙到 ±1，关节大幅乱动）。见故障报告阶段 C。

### 5.3 `asset_id` 写飞问题

若训练把 `asset_id` 设成**绝对路径形式的 repo_id**，保存逻辑可能变成：

```text
Path(ckpt/assets) / "/data1/..."   → 写到数据集目录，不进 ckpt
```

结果：推理在 ckpt 下找不到 stats → 回落 identity（若未手动拷贝真实 stats）。

**工程建议：**

1. 训练用短 `asset_id`，保证 `norm_stats.json` 落在 `ckpt/assets/`。
2. 启动后确认日志是「使用 checkpoint assets」而非「生成本地恒等统计」。
3. 打开 `ckpt/.../assets/**/norm_stats.json`，确认 `state.mean` 等不是全 0、`q01/q99` 不是 ±1`。
4. 长期：缺匹配 stats 应 **报错退出**，禁止静默 identity（当前实现仍会合成，属兼容行为）。

### 5.4 维匹配判定

`_norm_stats_match_action_dim`：

- 任一特征维存在且 ≠ `action_dim` → 不匹配。
- 至少一个特征维 == `action_dim` → 接受。

因此「只有 state 有维、actions 缺失」仍可能通过；完整 stats 应两者齐全。

---

## 6. Delta 动作与配置覆盖

### 6.1 训练侧语义

OpenPI `use_delta_joint_actions` + `delta_action_dims` 经 `make_bool_mask` 得到 mask：

```text
正整数 n：连续 n 维做 delta（监督 = action − state）
-1：该维保持绝对（通常夹爪）
```

TongBot 双臂训练应对齐：

```yaml
# 训练 cigai-agilex-train / 或推理 YAML 覆盖
delta_action_dims: [7, 7, -1, -1]   # 左7 delta | 右7 delta | 左爪 abs | 右爪 abs
```

库存 `pi05_aloha` 默认常为 Aloha `(6, -1, 6, -1)`（14D）。**只改推理 YAML 的 `openpi_action_dim: 16` 而不改 delta mask / 重训，行为未定义。**

推理 YAML：

```yaml
openpi_delta_action_dims: [7, 7, -1, -1]   # null = 不覆盖，用 TrainConfig 原值
```

`_apply_delta_action_dims_override` 在 TrainConfig 无该字段时只 warning，不改配置——需升级 vendored openpi training config。

### 6.2 推理时 delta→绝对

Policy 输出路径含 `AbsoluteActions`：

```text
cmd ≈ state + predicted_delta   （mask 为 True 的维）
```

因此：**训练时 `action − state` 的系统性偏差会在推理被「每步再加一遍」**。这是 J7 转圈的机制基础（§8）。

### 6.3 `adapt_to_pi`

| 值 | 含义 | TongBot |
|----|------|---------|
| `true` | AlohaInputs 做 π 关节约定重排/符号，并假设 14D | **禁止**（除非真用 14D Aloha 布局） |
| `false` | 按配置维原样进模型 | 双臂 16D / 左臂 8D 必选 |

`openpi_adapt_to_pi: true` 且 `openpi_action_dim != 14`：config_loader 与 `load()` 双处拦截。

---

## 7. YAML 与启动校验

### 7.1 双臂推荐片段

见 [`pi05_openpi_dual_arm_gripper_direct.yaml`](../config/lerobot/pi05_openpi_dual_arm_gripper_direct.yaml)：

```yaml
policy:
  type: pi05
  pi05_backend: openpi
  ckpt_dir: /path/to/ckpt
  openpi_config_name: pi05_aloha
  openpi_camera_names: [left, front, right]
  openpi_action_dim: 16
  openpi_action_horizon: 50
  openpi_adapt_to_pi: false
  openpi_delta_action_dims: [7, 7, -1, -1]
  keep_dims: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  rtc_enabled: false
```

左臂示例：`pi05_openpi_left_arm_direct.yaml` → 通常 `openpi_action_dim: 8`，`delta: [7, -1]`。

### 7.2 `config_loader` / `startup_validation` 会查什么

- `pi05_backend ∈ {lerobot, openpi}`
- openpi：`rtc_enabled` 必须 false
- `len(keep_dims) == openpi_action_dim`
- `adapt_to_pi` 与维数约束
- vendored `openpi` 目录可解析（`resolve_openpi_src_path`）

改字段后先跑：

```bash
python3 -m pytest infer/tests/test_startup_validation.py infer/tests/test_openpi_pi05_runtime.py -v
```

---

## 8. 坐标系、J7 ±120° 与 delta 交互（必读）

本节不是 runtime 内部逻辑，但 **改 runtime / 训推配置前必须理解**。细节与数值指纹见 [REPORT](../docs/REPORT_PI05_J7_SPIN_AND_INFER_ISSUES.md)。

### 8.1 两套角

| 名称 | 含义 |
|------|------|
| 逻辑角 | 外骨骼 / 策略 16D `action` 坐标系 |
| 机身角 | RealMan 实际下发与读回 |

控制约定（左右腕 J7，16D 下标 **6 / 13**；15D 控制向量下标另算）：

```text
下发：q_机身 = q_逻辑 + 120°
读回（推理 direct_robot_io）：q_逻辑 = q_机身 − 120°
```

实现位置：`infer/io/direct_robot_io.py`（`action_15d_to_tonggenesis_degrees` 对 command[9]/[16] `+= 120`；读状态对称 `-= 120`）。

### 8.2 数采落盘（修复前根因 / 现已对齐）

**修复前：**

```text
外骨骼 ──► 数据集 action（逻辑角；+120 未写入 parquet）
         └── bridge 下发时才 +120 ──► 机械臂

机械臂反馈 ──► 数据集 state（机身角；数采未做 −120）
```

于是训练集上：`action[J7] − state[J7] ≈ −120°`。OpenPI delta 会把该常数学成相对运动 → 推理转圈。

**现已修复：** `data_collect/robot_arm_bridge.py` 的
`rm75_arm_degrees_to_dataset_state_rad` 对左右 J7 **−120°** 后再落盘，与推理
`direct_robot_io` 一致。**旧数据集仍为修复前分布，须重采后重训 delta ckpt。**

ACT / LeRobot 绝对角 π₀.₅ 不把 `action−state` 当监督，对同一数采瑕疵不敏感。

### 8.3 与 runtime 的边界

| 层 | 该不该修 J7 |
|----|-------------|
| `openpi_pi05_runtime` | **否**。只保证 dim、norm、obs、infer 契约 |
| `direct_robot_io` / 数采 bridge | 已对齐：两边读状态 J7 −120、下发 +120 |
| 训练数据 + 重算 norm + **重训** | 正解；旧 delta ckpt 的 J7 不可靠 |
| 推理临时止血 | 锁 dim 6/13：`desired = current`；或试验去掉读状态 −120 |

**不要**在 runtime 里「智能」改 J7：会掩盖数据问题，且与 AbsoluteActions 双重补偿。

### 8.4 修 norm 后现象如何变化

| 阶段 | 现象 | 与 runtime 关系 |
|------|------|-----------------|
| 14D layout / 14D stats | 维错乱、广播报错 | `_accept_norm_stats` 应拦截 14D |
| Identity stats | 整体乱动 | 第 4 级回退；需换真实 stats |
| 真实 stats + 旧数据 | 非 J7 正常，J7 转圈 | runtime 工作正常；问题在数据×delta |

---

## 9. 与 LeRobot Pi05 / ACT 对照

| 项目 | OpenPI runtime | LeRobot Pi05 | ACT |
|------|----------------|--------------|-----|
| 权重 | openpi `params/` / safetensors | `pretrained_model/` | 各 backend |
| 归一化 | 外部/回退解析的 norm_stats | 常内嵌 ckpt | dataset stats |
| 动作监督 | 常为 **delta 关节** | 绝对角 | 绝对角 |
| RTC | 不支持 | 可选 | 无 |
| JAX | 必需 | 否 | 否 |
| 懒加载 | 是 | N/A | aloha 类似 |

同一 `Pi05Adapter`，`pi05_backend` 分流；改 backend 不要混用 ckpt 格式。

---

## 10. 开发与调试手册

### 10.1 本地改 runtime 的原则

1. **顶层只依赖 stdlib + numpy**，禁止顶层 `import openpi` / `jax`。
2. 图像预处理尽量纯 numpy/PIL，避免为 preprocess 拉起 JAX。
3. 覆盖 frozen TrainConfig 只用 `dataclasses.replace`，不要 mutate。
4. 任何「找不到 stats」路径必须打 **WARNING**，并写清来源字符串（便于对照日志）。
5. 维不匹配必须 skip，禁止静默广播。

### 10.2 启动日志检查清单

加载成功时确认类似：

```text
OpenPI sys.path 注入（vendored）: .../infer/policies
覆盖 TrainConfig.data.delta_action_dims: ... → (7, 7, -1, -1)   # 若 YAML 覆盖
加载 OpenPI policy: config=... action_dim=16 horizon=... adapt_to_pi=False delta_action_dims=(7, 7, -1, -1)
使用 checkpoint assets 归一化统计: ...     # 或明确回退/identity WARNING
OpenPI π₀.₅ 已加载
OpenPI Pi05 推理配置: chunk_size=... n_action_steps=... action_dim=16 ...
```

若出现「生成本地恒等统计」而训练并非 skip-norm：先停机查 assets。

### 10.3 真机异常对照

| 症状 | 优先查 |
|------|--------|
| `(16,) (14,)` | keep_dims / action_dim / 误用 14D stats（应已被拒；查旧代码或旁路） |
| 全身乱动、夹爪 ±1 | identity vs 真实 norm |
| 仅 L7/R7 匀速转，Δ≈−2.0 rad | 数采 J7 坐标系 × delta（§8） |
| adapt 相关索引错误 | `openpi_adapt_to_pi` 误开 |

Eval 日志可算 `mean(cmd − feedback)`；J7 贴近 −2.094 即指纹命中。

### 10.4 单测

[`infer/tests/test_openpi_pi05_runtime.py`](../tests/test_openpi_pi05_runtime.py)：

- 相机映射、HWC、resize-with-pad、path resolve
- norm：全无 → identity；14D stats + `action_dim=16` → 拒绝并合成 16D
- Mock policy：`build_observation` / `infer` 形状
- Adapter 分支：`pi05_backend=openpi` 不走 LeRobot from_pretrained

不 import 真实 jax；CI 可无 GPU。

### 10.5 常用命令

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
# conda activate openpi-pi05-infer

python3 -m pytest infer/tests/test_openpi_pi05_runtime.py -v

python3 infer/cli/run_batch_infer.py \
  --config infer/config/lerobot/pi05_openpi_dual_arm_gripper_direct.yaml
```

---

## 11. 相关文件索引

| 路径 | 角色 |
|------|------|
| `infer/policies/openpi_pi05_runtime.py` | 本模块 |
| `infer/policies/pi05_adapter.py` | 后端分流、建帧、chunk 步数 |
| `infer/policies/openpi/` | Vendored OpenPI |
| `infer/config_loader.py` | YAML → PolicyConfig（含 openpi_* 字段） |
| `infer/core/startup_validation.py` | 启动前契约检查 |
| `infer/io/direct_robot_io.py` | ±120°、UDP、init pose |
| `data_collect/robot_arm_bridge.py` | 数采下发 +120；读状态是否 −120 |
| `data_collect/modalities.py` | action/state 写入数据集 |
| `infer/config/lerobot/pi05_openpi_dual_arm_gripper_direct.yaml` | 双臂 16D 示例 |
| `infer/config/lerobot/pi05_openpi_left_arm_direct.yaml` | 左臂示例 |
| `infer/docs/OPENPI_ENV.md` | 环境与依赖 |
| `infer/docs/REPORT_PI05_J7_SPIN_AND_INFER_ISSUES.md` | J7 转圈与 norm 事故复盘 |
| `infer/policies/pi05_adapter.md` | Adapter 级流程（含 LeRobot） |

---

## 12. 变更时回归建议

改动 runtime 时至少覆盖：

1. **单测全绿**（上节）。
2. **维契约**：14↔16 拒收 stats；`adapt_to_pi` 非法组合报错。
3. **日志**：覆盖 delta / norm 来源字符串仍可读。
4. 若动到 norm 写回路径：用临时 ckpt 目录确认 `assets/` 落点，**禁止** `asset_id` 为绝对路径导致写飞。
5. 真机冒烟：非 J7 关节 `|cmd−fb|` 量级；若仍转圈，按 §8 / REPORT 处理，勿在 runtime 打补丁掩盖。
