# 数据采集（data_collect）

平台仓入口是仓库根 [`collect/`](../../collect/README.md)（本目录经软链暴露为 `collect/data_collect/`）。日常命令：

```bash
cd /path/to/data_Driven
python collect/run_collect.py --config collect/data_collect/config/left_arm_gripper_2cam.yaml
```

下面仍写模块细节。文中 `scripts/data_collect/` 是 cigai 旧路径，在本仓应读成 `collect/data_collect/`。

基于 YAML 配置的可复用数采模块，替代 legacy `new_data_collect_frame_test.py`。

## 目录结构

```text
collect/data_collect/   # Git 路径；实现在 infer/data_collect/
├── config/                          # YAML 配置
│   ├── default_full.yaml            # 双臂+夹爪+灵巧手+3相机
│   └── left_arm_gripper_2cam.yaml   # 左臂8维+2相机（Pi0.5/ACT 对齐）
├── run_collect.py                   # ★ 入口
├── collector.py                     # DataCollectApp / EpisodeSession
├── config_loader.py                 # YAML → CollectConfig
├── modalities.py                    # 模态定义与向量拼装
├── sync.py                          # 时间戳缓冲与同步引擎
├── udp_sources.py                   # UDP 接收
├── camera_source.py                 # RealSense 封装
├── dataset_writer.py                # LeRobot v3.0 写盘
├── frame_builder.py                 # 同步样本 → 帧
└── logging_setup.py
```

## 快速开始

```bash
conda activate lerobot   # 或已安装 pyrealsense2 + lerobot 的环境
cd /path/to/data_Driven
export PYTHONPATH="$(pwd)/legacy/CQTongBot_infer/src:$(pwd)/infer:${PYTHONPATH:-}"

# 编辑 collect/data_collect/config/*.yaml 中的 dataset.root、network、cameras.serials
python collect/run_collect.py --config collect/data_collect/config/left_arm_gripper_2cam.yaml
```

下位机流程与 README 模块一相同（hand_read → new_exo_control → exo_read）。

## 配置说明

### 机械臂模态（modalities.state / modalities.action）

| 键 | 含义 | UDP 来源 |
|----|------|----------|
| `left_arm` | 左臂 7 关节 | robot_state/action [0:7] |
| `right_arm` | 右臂 7 关节 | robot_state/action [7:14] |
| `left_gripper` | 左夹爪 1 维 | robot_state/action [14] |
| `right_hand` | 右灵巧手 8 维 | hand_state/action [0:8] |

每个模态三个开关：

- `enabled`: 是否写入数据集
- `required`: 同步失败是否丢弃整帧
- `fallback`: `drop_frame` | `hold_last`（仅建议用于 optional 的 right_hand）

### 相机（cameras.devices）

列表顺序 = `camera_module` 取流顺序 = `images[i]` 索引。
`key` 为 LeRobot feature 名，如 `observation.images.cam_left`。

### 时间同步（sync.clock_offset）

若 `[SYNC]` 日志显示图像系统性滞后，调整：

```yaml
sync:
  clock_offset:
    camera: -0.67   # 示例，以实机 [SYNC] 为准
```

## 相对 legacy 脚本的改进

1. YAML 配置模态与相机，无需改代码
2. LeRobot **v3.0** API（`lerobot.datasets`）
3. 修正 Episode 计数逻辑
4. 同步统计按控制 tick 计算成功率与有效帧率
5. optional 模态 `hold_last` 降级，减少整帧丢弃
6. 模块化 Class，便于单测与复用

## 采后 QA（evaluation.py）

`scripts/evaluation.py` 在仓库中为**离线 CLI**（见 README §离线评估），legacy 采集不会自动调用。
`data_collect` 提供可选钩子，**默认关闭**（`qa.enabled: false`）。

启用示例：

```yaml
qa:
  enabled: true
  threshold: 0.65              # 综合分及格线
  warn_if_below_threshold: true  # 低于阈值打 WARN 日志
  save_report: true
  report_subdir: qa_reports      # 相对 dataset_root/<repo_id>/
  report_path: null              # 或绝对路径目录
```

每次 `save_episode` 成功后评估**刚保存的那一集**，报告 JSON 写入
`{dataset_root}/{repo_id}/qa_reports/qa_epXXX_*.json`。

手动全量扫描仍可用：

```bash
python scripts/evaluation.py --dataset-root ... --repo-id ...
```

## 编程式调用

```python
from data_collect import load_collect_config, DataCollectApp

cfg = load_collect_config("collect/data_collect/config/left_arm_gripper_2cam.yaml")
app = DataCollectApp(cfg)
app.run_interactive()
```


## 实时夹爪曲线刷新

`realtime_gripper_plot.update_interval_sec: 0.0` 表示每生成一帧采集数据就立即刷新 OpenCV 曲线窗口，不再额外限频。实际曲线更新频率由 `dataset.fps` 和有效同步帧率决定。


## 最新下位机协议适配（2026-07）

- `18131 robot_action`: 16 个 float64，内容为 `14关节 + 左外骨骼夹爪 + timestamp`；采集端去掉末尾时间戳后得到 15D action。
- `30016 gripper_state`: 3 个 float64，内容为 `真机左夹爪 + 真机右夹爪 + timestamp`。
- `18140 exo_state`: 20 个 float64，用于诊断，内容为 `17外骨骼关节 + 左夹爪 + 右夹爪 + timestamp`。
- `30026 exo_command`: Episode 开始自动发送 `1.0`，结束自动发送 `0.0`。
- 当前 `observation.state` 为 16D（14关节兼容取自 action + 两个真机夹爪）；`action` 为 15D（14关节 + 左外骨骼夹爪）。最新下位机没有右夹爪 action。

## 显示性能开关

所有配置文件均支持以下开关，且默认开启：

```yaml
cameras:
  enabled: true
  enable_display: true      # false：仍采集相机，但不弹出实时画面

realtime_gripper_plot:
  enabled: true             # false：不创建夹爪曲线窗口
```

相机显示使用 **OpenCV** 窗口（与夹爪曲线同栈），启动时创建一次，跨 Episode 常驻；结束单个 Episode 不关窗口，退出采集程序时关闭。手动关掉某路相机窗口后，该路不再重建。

## RM75 observation.state 与夹爪来源配置

当前版本中，`observation.state` 的机械臂关节由上位机通过 RealMan SDK 调用
`rm_get_current_arm_state()` 读取，保存顺序为左臂 7 维、右臂 7 维，统一为弧度。
**左右臂第 7 关节在落盘前会 −120°**（`rm75_arm_degrees_to_dataset_state_rad`），
与下发时的 +120°、推理 `direct_robot_io` 读状态 −120° 对齐，使
`action`/`state` 同为逻辑角。旧数据（修复前）state 仍为机身角，不可直接用于
OpenPI delta 训练。

所有配置文件均包含：

```yaml
modalities:
  # 0：数据集和实时曲线中的夹爪字段来自真实夹爪 UDP 30016
  # 1：数据集和实时曲线中的夹爪字段来自外骨骼 UDP 18131 的 robot_action[14]
  #    当前外骨骼协议只有一个左夹爪值，因此左右夹爪字段都会复制该值。
  gripper_source: 0

  # 写入数据集和实时绘图前的夹爪值过滤。
  # 语义：1.0=全开/未按，0.0=闭合/按下。
  # 未按压时真实夹爪/外骨骼偶发周期波动会被吸附为 1.0。
  gripper_value_filter:
    enabled: true
    open_value: 1.0
    open_deadband: 0.08
    close_threshold: 0.85
    stable_samples: 2
    output_mode: binary

robot_arm:
  enabled: true
  state_hz: 30.0

  # 真实夹爪控制仍然始终使用外骨骼夹爪 robot_action[14]，但会先经过防抖过滤。
  gripper_filter:
    enabled: true
    open_value: 1.0
    open_deadband: 0.06
    change_threshold: 0.08
    stable_samples: 2
    ema_alpha: 1.0
```

### 夹爪源头说明

系统里有三类“夹爪值”，不要混淆：

| 名称 | 网络来源 | 含义 | 主要用途 |
|---|---|---|---|
| 外骨骼夹爪目标值 | 下位机 `exo_read_safety.py` → UDP `18131` → `robot_action[14]` | 操作者在外骨骼上捏出的目标开合比 | 控制真实夹爪；也可作为数据集/曲线夹爪字段 |
| 真实夹爪反馈值 | 下位机 `only_gripper.py` → UDP `30016` → `gripper_state[0:2]` | Dynamixel 真实左右夹爪实际位置 | 默认写入数据集和实时曲线 |
| 夹爪控制指令 | 上位机 `RobotArmBridge` → UDP `30015` → `only_gripper.py` | 发给真实夹爪电机的控制命令 | 实际驱动左右真实夹爪 |

`modalities.gripper_source` 只控制“写入数据集/实时绘图时夹爪字段来自哪里”：

| 配置 | `observation.state` 中启用的夹爪 | `action` 中启用的夹爪 | 实时夹爪曲线 | 是否需要 UDP `30016` |
|---:|---|---|---|---|
| `0` | 真实夹爪 `gripper_state[0:2]` | 真实夹爪 `gripper_state[0:2]` | 真实夹爪反馈曲线 | 需要 |
| `1` | 外骨骼夹爪 `robot_action[14]` | 外骨骼夹爪 `robot_action[14]` | 外骨骼夹爪目标曲线 | 不需要 |

注意：`gripper_source` 不改变真实夹爪的控制来源。真实夹爪的控制始终来自外骨骼 `robot_action[14]`，经过 `robot_arm.gripper_filter` 防抖过滤后，通过 UDP `30015` 发给下位机 `only_gripper.py`。

推荐配置是：

```yaml
modalities:
  gripper_source: 0
```

这样保存的数据语义最清楚：

```text
observation.state 夹爪 = 真实夹爪实际位置
action 夹爪            = 当前配置下也保存真实夹爪实际位置
真实夹爪控制命令       = 外骨骼夹爪目标值 robot_action[14]，经过防抖过滤后下发
```

如果后续希望严格模仿学习语义，也可以改成：

```text
observation.state 夹爪 = 真实夹爪反馈
action 夹爪            = 外骨骼夹爪目标
```

但当前版本是按照 `gripper_source` 统一控制 state/action 中所有已启用夹爪字段。


### 夹爪曲线为什么未按压固定为 1.0

当前版本在 `FrameBuilder` 组帧前增加了 `modalities.gripper_value_filter`。它作用于最终写入 `observation.state`、`action` 和实时绘图的夹爪字段。

语义固定为：

```text
1.0 = 全开 / 未按夹爪
0.0 = 闭合 / 按下夹爪
```

过滤规则：

- 靠近 `open_value=1.0` 的值直接吸附为 `1.0`；
- 高于 `close_threshold=0.85` 的中间波动视为空闲噪声，也保持 `1.0`；
- 只有连续 `stable_samples=2` 帧低于 `close_threshold`，才认为是真的按压；
- `output_mode: binary` 时，最终保存和绘图只有 `1.0` 或 `0.0` 两个值。

因此，没有按夹爪时曲线应该是一条稳定的 `1.0` 直线；只有真正持续按下时才变为 `0.0`。

### 当前完整字段来源

| 字段 | 来源 | 说明 |
|---|---|---|
| `observation.state[0:7]` | RM75 左臂 SDK → **J7−120°** → rad | 逻辑角，与 action 对齐 |
| `observation.state[7:14]` | RM75 右臂 SDK → **J7−120°** → rad | 逻辑角，与 action 对齐 |
| `observation.state` 中夹爪 | 由 `gripper_source` 决定 | `0` 为真实夹爪，`1` 为外骨骼夹爪 |
| `action[0:7]` | UDP `18131` `robot_action[0:7]` | 左外骨骼目标关节，单位 rad |
| `action[7:14]` | UDP `18131` `robot_action[7:14]` | 右外骨骼目标关节，单位 rad |
| `action` 中夹爪 | 由 `gripper_source` 决定 | `0` 为真实夹爪，`1` 为外骨骼夹爪 |
| 真实夹爪控制 | UDP `18131` `robot_action[14]` → 防抖 → UDP `30015` | 与 `gripper_source` 无关 |

该功能仅修改上位机。下位机已经同时提供外骨骼夹爪和真实夹爪协议，无需修改。

## RGB 与 Depth 对齐及 LeRobot v3 元数据

开启深度采集时，配置统一使用：

```yaml
cameras:
  depth:
    enabled: true
    format: png
    align_to_color: true
    feature_prefix: observation.depth
```

每台 RealSense 的 RGB 与 depth 从同一个 `frameset` 获取，并通过
`rs.align(rs.stream.color)` 使用相机标定参数把 depth 重投影到 RGB/color
像素坐标系。程序会检查对齐后的 depth 高宽必须与 RGB 一致；不使用
`cv2.resize`，避免破坏深度几何意义。

深度以 16-bit PNG 独立保存，字段名示例：

- `observation.depth.cam_right`
- `observation.depth.cam_left`
- `observation.depth.cam_mid`

这些字段会写入 `meta/info.json`，并记录 `dtype=uint16`、shape、对应 RGB
字段、对齐方式、相机序列号和 PNG 路径模板。实际文件位于：

```text
depth/observation.depth.cam_right/chunk-000/episode-000000/frame-000000.png
```

注意：这里保证的是“同一台相机的 RGB-depth 对齐”。多台独立 RealSense
之间仍是主机时间近似同步，不等同于硬件触发同步。

### 夹爪按下响应延迟修正

旧版本的夹爪控制防抖使用“数值稳定确认 + EMA 平滑”：外骨骼从 `1.0` 按到 `0.0` 的过程中，原始值会连续变化，稳定计数容易反复清零；确认以后又要经过 EMA 从 `1.0` 慢慢降到闭合值，所以真实夹爪会明显滞后。

当前版本改为“阈值连续确认”：

- 未按压/松开回到全开区间时，立即发送 `1.0`；
- 单帧低值脉冲仍会被过滤；
- 连续 `stable_samples=2` 帧低于闭合阈值后，立即把当前外骨骼夹爪值发送给真实夹爪；
- 默认 `ema_alpha: 1.0`，不再慢速平滑闭合指令。

在 `control_hz: 30.0` 时，`stable_samples: 2` 的确认时间约为 `2 / 30 = 0.067s`。如果仍然觉得慢，可以把 `robot_arm.gripper_filter.stable_samples` 改为 `1`；如果误触发增多，再改回 `2`。
