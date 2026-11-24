# 推理客户端（infer 包）

本目录是 **data_Driven** 里的真机客户端：模型加载适配、相机同步、机械臂/夹爪、日志、视频、任务分段和离线评估。日常开跑用仓库根 [`recipes/`](../../recipes/README.md)，不要从 cigai 旧路径抄命令。

```bash
cd /path/to/data_Driven
export PYTHONPATH="$(pwd)/legacy/CQTongBot_infer/src:$(pwd)/infer:${PYTHONPATH:-}"
```

总览：[../../README.md](../../README.md)。真机表：[../../docs/gpu_master_infer.md](../../docs/gpu_master_infer.md)。

---

# CQTongBot 推理与记录说明

本目录负责模型加载、相机同步、机械臂/夹爪控制、日志记录、视频保存、任务分段和离线评估。

## 目录结构

```
infer/
├── cli/           # 命令行入口（run_infer / run_batch_infer / run_eval）
├── config/        # YAML 实验配置（lerobot/ / aloha/，见 config/README.md）
├── core/          # 引擎主循环（engine、session、startup_validation）
├── io/            # 硬件 I/O（机械臂 SDK、相机、夹爪 UDP、初始位姿）
├── exo/           # 外骨骼数采与安全
├── ui/            # 终端交互（脚踏板、prompt）
├── batch/         # 批量消融（OOD、续跑、远程同步）
├── runs/          # 产物目录完整性检查
├── policies/      # 模型 adapter 注册
├── runners/       # native / legacy / batch runner
├── recording/     # 日志、视频、健康监控
├── reports/       # 离线评估报告
├── metrics/       # 评估指标
├── tests/         # 单元测试
└── docs/          # 补充文档（如 BATCH_INFER.md）
```

根目录仍保留 `run_infer.py`、`run_batch_infer.py` 等薄 shim，兼容旧命令。

## PICO 输入诊断

在已安装 PyZMQ 的环境中，从 `VLA-INFER/tk_data_Driven_dev/infer/` 运行。发布端使用同机的 `robot_snapshot_20261118/vr_pub.py`。

```bash
python -m infer.io.pico_input --endpoint tcp://127.0.0.1:5555 --timeout 0.3
```

诊断程序只订阅 PICO 数据。它分别报告追踪可用、无数据、设备时间戳冻结、坏帧和序号或时间戳回退。

## RM75 推理中的 PICO 接管

第一版面向 `native` 异步推理和 `direct_robot.enabled: true` 的 RM75 双臂。默认关闭。上位机使用已有的 PyZMQ、SciPy 和 RM SDK 环境。

```yaml
pico_recovery:
  enabled: true
  endpoint: tcp://127.0.0.1:5555
  timeout_sec: 0.3
  # 可选。按台架校准填写左右夹爪的开合比，两项同时设置。
  # gripper_open: [左开口比, 右开口比]
  # gripper_closed: [左闭合比, 右闭合比]
```

- **X 接管**。清除旧动作块并请求双臂停止。以当前实测关节和 PICO 位姿建立相对参考，双手分别控制对应机械臂。
- **按住扳机闭合，松开张开**。配置开合比后启用。接管时已按住的扳机要先松开再按。另一侧夹爪维持本次接管的目标或实测开口。
- **Y 原地交回**。停止手动跟随，将策略状态重置排在正在执行的旧请求之后，并等待交回后的新相机和机器人状态。首个模型关节目标与实测相差超过 8° 时回到保持。
- **X + Y 保持**。追踪冻结、断流、无效帧或设备健康暂停也会退出跟随。追踪恢复后需重新按 X 接管或按 Y 交回。PICO 持有控制权时暂停处理冲突的外骨骼和任务切换事件。

手动动作的接管点位移半径为 10 cm，腕部相对旋转上限约 26°，单次关节步长上限为 1°。双臂碰撞和现场跟随效果仍待台架验证。普通推理、外骨骼、PICO 共用控制链路，启用此功能的同一套机械臂应由当前推理进程独占发臂。控制切换、目标关节和人工夹爪命令写入推理流程日志。

## 文档索引

| 文档 | 内容 |
|------|------|
| [docs/BATCH_INFER.md](docs/BATCH_INFER.md) | 批量消融操作手册（实习生） |
| [docs/README.md](docs/README.md) | 本目录全部文档索引 |
| [config/README.md](config/README.md) | 实验配置索引（lerobot / aloha） |
| [policies/README.md](policies/README.md) | Policy 注册与推理链路 |
| [../tools/README.md](../tools/README.md) | 共用工具（policy / camera / dataset） |
| [../../docs/FORKS.md](../../docs/FORKS.md) | 训练 fork（不在本目录） |
| [../vla_eval/README.md](../vla_eval/README.md) | VLA 自动评估 |

默认推荐使用 `infer/config/lerobot/pi05_left_arm_direct.yaml`：

- 上位机直接通过 RealMan SDK 控制机械臂。
- 相机由推理进程本地直读 RealSense。
- 夹爪仍通过 UDP 发给下位机 `lower_machine/only_gripper.py`。
- 支持同一个**组目录**内连续采集多个任务段（如 `action-step_10/`、`ood-language/`）。
- 支持脚踏板触发任务热键（默认 `Ctrl+.`，可在配置 `pedal.task_toggle` 修改）写入视频时间戳 marker。
- 推理时实时显示 `cam_left` / `cam_mid` 相机窗口（直读帧）。
- **设备健康实时界面**（`device_health` 窗口 + 相机顶栏）：下位机夹爪 UDP 断联、夹爪指令与实测开口连续超阈、ping 失败、状态/相机流中断等重大异常会**即刻暂停推理发臂**，恢复后需确认（默认 `pedal.confirm` / `Y`）再继续。

## 常用入口

```bash
cd /path/to/data_Driven
export PYTHONPATH="$(pwd)/legacy/CQTongBot_infer/src:$(pwd)/infer:${PYTHONPATH:-}"
cd infer
python3 infer/cli/run_infer.py --config infer/config/lerobot/pi05_left_arm_direct.yaml
```

真机 π₀.₅ 用 `recipes/pi05/start_pi05_infer.sh`，不要直接抄上面这条旧单臂配置。

如果要离线评估已有结果：

```bash
python3 infer/cli/run_eval.py --test-dir /path/to/results/action-step_10
```

### 推理代码版本号（含远程无 git）

每次推理会把代码版本写入 `meta.json`（字段 `infer_git_commit` / `infer_git_source`）。读取顺序：

1. 环境变量 `INFER_GIT_COMMIT`（可选 `INFER_GIT_BRANCH`、`INFER_GIT_DIRTY`）
2. 本机 `git`（有 `.git` 时；成功会自动刷新 stamp）
3. `infer/INFER_GIT_REVISION.json`（随代码 rsync 到远程）

rsync 常排除 `.git/`，同步到上位机前在有仓库的机器上盖章一次：

```bash
python3 -m infer.scripts.stamp_git_revision
```

## 推荐启动流程

下位机全部代码见 [lower_machine/README.md](../lower_machine/README.md)。

### 1. 启动下位机（夹爪）

`direct_robot.gripper_mode: udp` 时，推理进程把夹爪指令发到 `network.low_host:gripper_port`，需先在**下位机** (192.168.1.13) 运行：

```bash
cd lower_machine
python3 run.py --gripper-only
```

该进程接收夹爪控制 UDP（30015），并通过 **30016** 把 Dynamixel 实测夹爪位置推回上位机；不控臂。

上位机 `network.gripper_state_port`（默认 30016）须与下位机一致；观测 state 优先用 UDP 实测，无包时保持缓冲内旧值（不用下发指令缓存）。

### 2. 启动推理

在上位机运行：

```bash
cd /home/cigai-24-3/infer
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 infer/cli/run_infer.py --config infer/config/lerobot/pi05_left_arm_direct.yaml
```

推理启动后会：

- 加载模型一次。
- 本地打开 RealSense 相机。
- 连接 RealMan SDK 读取状态并发送机械臂动作。
- 向夹爪接收端发送 UDP 夹爪命令。
- 打开 `cam_left` / `cam_mid` 实时显示窗口（若 `session.visualize_cam_*` 为 true）。
- 开始写入日志、视频和轨迹数据。

### 3. 外骨骼安全（可选）

`exo_safety.enabled: true` 时下位机需完整启动（非 `--gripper-only`）：

```bash
cd lower_machine
python3 run.py
```

上位机 Ctrl+1/Ctrl+2 控制 exo 上传与接管；详见 [EXO_SAFETY_README.md](../EXO_SAFETY_README.md)。

## 配置文件

实验配置按训练/推理框架分子目录，完整索引见 [config/README.md](config/README.md)。

本机路径、IP、相机序列号和串口优先放在：

```text
infer/config/local_env.yaml
```

实验相关参数放在具体配置里，例如：

```text
infer/config/lerobot/pi05_left_arm_direct.yaml      # 左臂 + 左夹爪（默认）
infer/config/lerobot/pi05_right_arm_direct.yaml     # 右臂 + 夹爪
infer/config/lerobot/pi05_dual_arm_gripper_direct.yaml  # 双臂 + 左夹爪
infer/config/lerobot/pi05_left_arm.yaml
infer/config/lerobot/act_left_arm_direct.yaml
infer/config/lerobot/act_left_arm.yaml
infer/config/aloha/act_aloha_left_arm_direct.yaml   # 原生 ALOHA ACT
```

### Direct 配置文件一览

| 场景 | 配置文件 | `keep_dims` | OOD presets |
|------|----------|-------------|-------------|
| 左臂 + 左夹爪 | `lerobot/pi05_left_arm_direct.yaml` | 8 维（左臂 7 + 左夹爪 1） | `infer/config/ood_language_presets.yaml` |
| 右臂 + 夹爪 | `lerobot/pi05_right_arm_direct.yaml` | 8 维（右臂 7 + 夹爪 1） | `infer/config/ood_language_presets.yaml` |
| 双臂 + 双夹爪 | `lerobot/pi05_dual_arm_gripper_direct.yaml` | 16 维（左臂 7 + 右臂 7 + 左夹爪 1 + 右夹爪 1） | `infer/config/ood_language_presets_dual_arm.yaml` |
| LeRobot ACT 左臂 | `lerobot/act_left_arm_direct.yaml` | 8 维 | 无 OOD |
| 原生 ALOHA ACT 左臂 | `aloha/act_aloha_left_arm_direct.yaml` | 8 维 | 无 OOD |
| 原生 ALOHA DP 左臂 | `aloha/dp_aloha_left_arm_direct.yaml` | 8 维 | 无 OOD |
| 原生 ALOHA RDT 左臂 | `aloha/rdt_aloha_left_arm_direct.yaml` | 8 维 | 无 OOD |
| DP 左臂 | `lerobot/dp_left_arm_direct.yaml` | 8 维 | `infer/config/ood_language_presets.yaml` |

原生 ALOHA（TongBot）固定 `arm_joint_state_dim: 8`；环境安装见 [docs/ALOHA_ENV.md](docs/ALOHA_ENV.md)。

`policy.ckpt_dir` 与 `policy.keep_dims` 维数必须一致；双臂模型须用 16 维 checkpoint。各 direct 配置的 `batch_infer.vla_auto_eval` 默认开启：每组 action-step 结束后在 `eval_table/` 生成 `eval_result.et` 与 `summary.md`（详见 [docs/BATCH_INFER.md](docs/BATCH_INFER.md)）。

批量推理启动示例：

```bash
# 双臂 + 左夹爪
python3 infer/cli/run_batch_infer.py --config infer/config/lerobot/pi05_dual_arm_gripper_direct.yaml

# 单次推理
python3 infer/cli/run_infer.py --config infer/config/lerobot/pi05_dual_arm_gripper_direct.yaml
```

实验 YAML 中 `local_env: ../local_env.yaml` 会把本机配置深度合并到主配置中。建议不要把 `task_prompt`、消融参数等实验内容放进 `local_env.yaml`。

## 关键配置

### direct 机械臂控制

```yaml
direct_robot:
  enabled: true
  sdk_path: ~/RM_API2-main
  gripper_mode: udp
  gripper_initial_position: [0.7, 0.7]
  robot_r_host: "192.168.1.17"
  robot_l_host: "192.168.1.18"
  state_poll_hz: 100.0
```

`gripper_mode: udp` 表示机械臂直连控制，但夹爪仍发送 UDP 到下位机 `lower_machine/only_gripper.py`。

### 相机

```yaml
camera_sender:
  enabled: true
  transport: local
  devices:
    # devices[0]=cam_left（左腕 f1480854）, devices[1]=cam_mid（胸前 f1420947）
    - serial: "f1480854"
      fps: 30
    - serial: "f1420947"
      fps: 30
```

`transport: local` 表示推理进程直接打开 RealSense，不需要单独运行旧的 `camera_module`。

启动校验会读取 checkpoint `config.json` 的 `input_features` 相机 shape，自动设置 RealSense 采集分辨率，与训练保持一致，无需在 YAML 里手写 width/height。

### 连续任务模式

```yaml
session:
  fps: 15
  max_duration_sec: 30.0
  prompt_continue: true
  max_tests: null
  task_mode: continuous_tasks
```

该模式会在同一个**组目录**里连续采集多个任务段：

- 每个任务最多运行 `max_duration_sec` 秒。
- 到时后在终端询问是否开始下一个任务。
- 模型不重新加载。
- 视频不停止。
- 日志不停止。
- 任务开始/结束边界写入 `task_segments.jsonl`。

### 打点后即时追问人工评测

按 `Ctrl+.`（或脚踏板 `task_toggle`）打点之后，终端会立刻追问这次推理的**文字描述**：

```
[EVAL] 刚按 Ctrl+. 标记：推理 #473（marker #20）—— 请用文字描述这次结果
[EVAL]   请用文字描述本次表现（例如：夹爪提前闭合、轨迹发抖、物体没夹稳）  提示：可对照 PI0.5 的结果作为参考。
[EVAL]   输入描述后回车进入下一步；空行=跳过。无超时，会一直等待输入。
```

- 默认**没有超时**：会一直等到你输入描述后回车，再进入下一条。
- 不给 1/2/3 选项，直接写描述；答案写 `inference_eval_marks.jsonl`，一行一次打点；同条带上 marker 编号、**这次推理的帧号**和分段耗时（`model_sec/ik_sec/flow_sec/network_sec`…）。
- 追问跑在独立线程：不阻塞推理控制循环，也不影响其它热键；多按几次会排队，一次只问一个。
- **没回答也照样落一条**（`skipped` / `no_stdin` / `aborted` / `burst`），并且只有显式写出的 `success/partial/fail` 进成功率分母——文字描述不进成功率，不会把"没评测"读成"0% 成功"。
- 后台 `nohup`（stdin 非终端）时第一条记 `no_stdin`，之后不再打扰。
- 组结束时汇总写 `inference_eval_summary.json`，并进 `inference_summary.json` 的 `human_eval` 段与实验索引 `metrics.human_eval`。

关闭或（可选）打开超时：

```yaml
session:
  eval_prompt_on_marker: false   # 默认 true
  eval_prompt_timeout_sec: 8     # 默认 0=一直等待；>0 才超时记 timeout
```

如果设置：

```yaml
session:
  max_duration_sec: null
```

则当前组目录会一直运行，直到手动停止或程序异常退出。

### 相机实时显示

```yaml
session:
  visualize_cam_left: true
  visualize_cam_mid: true
  cam_left_window_name: cam_left
  cam_mid_window_name: cam_mid
```

推理时会弹出 OpenCV 窗口显示 `cam_left`、`cam_mid` 的**直读帧**（非推理对齐帧）。若无法打开 GUI，会打印 warning 并关闭显示，不中断推理。

### 外骨骼安全（exo_safety）

```yaml
exo_safety:
  enabled: false          # true 时启用 Ctrl+1/2 流程
  exo_state_port: 18140
  exo_command_port: 30026
  # record_start_delay_sec: 2.0           # 选数采后延迟开录；先弹中间相机窗口
  # collect_cam_mid_window_name: exo_collect_cam_mid

pedal:
  exo_takeover: ctrl+1    # 下发 START，接管，询问是否数采
  exo_safety_infer: ctrl+2
```

默认 `enabled: false`，关闭时对现有推理零影响。详见 [EXO_SAFETY_README.md](../EXO_SAFETY_README.md)。

选数采后：中间相机预览窗口立即弹出（不播放）→ `record_start_delay_sec`（默认 2s）后开始写帧并同步播放 → Ctrl+2 结束数采时关闭该窗口。

`align_enabled: true`（默认）时接管流程为：

1. Ctrl+1 → 对齐阶段（臂 hold）
   - `align_mode: auto`（默认）：下位机主动驱动外骨骼到机械臂位姿（**请勿穿戴/对抗**）
   - `align_mode: manual`：仅上报，人摆外骨骼
2. 关节误差达标（`align_max_joint_err_deg` / `align_rms_joint_err_deg`）
3. 按 `y` 正式接管（不会自动开始）；未达标会拒绝
4. 再询问是否数采；Ctrl+2 结束接管

`align_enabled: false` 恢复旧行为：Ctrl+1 立即跟人并询问数采。

自动对齐需同步更新下位机 `exo_read_safety.py` / `TG_exo`（指令 `3=START_AUTO_ALIGN`，目标端口 `30027`）。若 `MultiDynamixel` 写角接口名不匹配，下位机日志会打印可用方法列表。

### 脚踏板 marker

推理时按任务热键（默认 `Ctrl+.`，见配置 `pedal.task_toggle`）会记录当前视频时间戳 marker。脚踏板映射为同一组合键即可。

marker 写入：

```text
组目录/keyboard_markers.jsonl
```

每条记录包含：

- `wall_time`
- `iso_time`
- `latest_camera_timestamps`
- `video_frame_counts`
- `trajectory_rows`
- `inference_frame_index`
- `task_segment_index`
- `source`

推理进程优先使用 `pynput` 全局监听任务热键（默认 `pedal.task_toggle: ctrl+.`）。如果 `pynput` 不可用，会在非连续任务场景退回终端 stdin 监听。连续任务模式需要终端输入确认，因此不会使用 stdin fallback 抢输入。

可以用旁路脚本检查脚踏板实际发出的按键：

```bash
python3 pedal_monitor.py
```

按 `ESC` 可退出该监听脚本。

## 输出目录

批量推理每组产物**直接**写在组目录下（如 `action-step_10/`、`ood-language/`），包含视频、日志、轨迹、评估等完整包。

```text
action-step_10/
├── meta.json
├── inference_params.json
├── inference_summary.json
├── run.log
├── machine_flow.jsonl
├── alerts.jsonl
├── eval_log.jsonl
├── eval_images/
├── eval_videos/
│   └── frame_timestamps.jsonl
├── videos/
│   ├── cam_left_wrist.mp4
│   ├── cam_left_wrist_depth.mp4
│   ├── cam_mid_chest.mp4
│   ├── cam_mid_chest_depth.mp4
│   └── frame_timestamps.jsonl
├── trajectory_data/
│   └── trajectory.parquet
├── task_segments.jsonl
├── keyboard_markers.jsonl
├── inference_eval_marks.jsonl      # Ctrl+. 打点后的人工评测答案
├── inference_eval_summary.json     # 上述答案的汇总（成功率/各档计数）
├── exo_mode_markers.jsonl          # exo_safety 启用时
├── （LeRobot 数据集见 exo_safety.collect_config 的 dataset.root，Ctrl+2 save_episode）
├── evaluation_report.json
├── evaluation_report.txt
└── evaluation_report.csv
```

### 每个文件是干什么的

| 路径 | 主要内容 | 什么时候看 |
|------|----------|------------|
| `meta.json` | 本次推理的基本信息：实验名、模型路径、提示词、fps、engine、启动时间、**推理代码 git 提交号**等 | 想确认这次用了哪个模型、哪个提示词、哪版推理代码 |
| `inference_params.json` | 和 `meta.json` 类似，保存推理参数快照（含 git 提交） | 复现实验配置时看 |
| `inference_summary.json` | 推理结束摘要：总时长、停止原因、chunk 参数、debug 图路径等 | 想快速知道这次为什么结束 |
| `run.log` | 人能直接读的运行日志：同步状态、每帧推理耗时、动作发送摘要、warning 等 | 先看这个排查问题 |
| `machine_flow.jsonl` | 机器可读的详细流程日志：同步失败、chunk 推理、动作发送、marker、任务边界等 | 做自动分析或画 debug 图 |
| `alerts.jsonl` | 告警日志：同步异常、模型输出异常、推理 fatal 等 | 推理出问题时看 |
| `eval_log.jsonl` | 每次模型推理的完整记录：输入图像路径、输入 state、模型输出 action chunk、执行反馈 | 想检查模型到底输出了什么 |
| `eval_images/` | `eval_log.jsonl` 引用的逐帧相机图片 | 想逐帧看模型输入图像 |
| `eval_videos/` | eval 视角的视频，通常按模型推理帧写入 | 想看和 `eval_log` 对应的视频 |
| `eval_videos/frame_timestamps.jsonl` | `eval_videos/` 中每一帧对应的推理帧和时间戳 | 对齐 eval 视频与日志 |
| `videos/` | 训练/后处理更常用的视频文件 | 想看整段录制视频 |
| `videos/cam_left_wrist.mp4` | 左腕相机 RGB 视频 | 看夹爪附近细节 |
| `videos/cam_left_wrist_depth.mp4` | 左腕深度视频（JET 伪彩，与 RGB 对齐） | 看距离/遮挡 |
| `videos/cam_mid_chest.mp4` | 胸前/中间相机 RGB 视频 | 看任务整体过程 |
| `videos/cam_mid_chest_depth.mp4` | 中相机深度视频（JET 伪彩） | 看场景深度 |
| `videos/frame_timestamps.jsonl` | `videos/` 中每一帧的视频帧号和时间戳 | 后期裁剪视频、对齐轨迹 |
| `trajectory_data/trajectory.parquet` | 可训练用的轨迹数据：机器人 state、模型 action、时间戳、任务段编号 | 做训练数据提取时最重要 |
| `task_segments.jsonl` | 连续任务模式下，每个任务开始/结束的时间戳和视频帧计数 | 从长视频中切出每个任务 |
| `keyboard_markers.jsonl` | 按任务热键（默认 Ctrl+.）或脚踏板时写入的人工 marker | 人工标记关键时刻 |
| `inference_eval_marks.jsonl` | 每次打点后追问"这次推理评测结果"的人工答案：判定（成功/部分完成/失败/跳过/未答）+ 备注 + **同条附带这次推理的分段耗时**（model/ik/flow/network…） | 把人工判据和机器指标配成一对 |
| `inference_eval_summary.json` | 上面这些判定的汇总（`asked` / `answered_valid` / 各档计数 / `success_rate`） | 汇报某次 run 的人工成功率 |
| `evaluation_report.json` | 自动评估报告，机器可读 | 程序分析 |
| `evaluation_report.txt` | 自动评估报告，人可读摘要 | 快速看结果 |
| `evaluation_report.csv` | 自动评估报告，表格格式 | Excel/表格分析 |

### 最常用的几个文件

如果只是回看视频：

- `videos/cam_mid_chest.mp4`
- `videos/cam_left_wrist.mp4`
- `videos/cam_mid_chest_depth.mp4` / `videos/cam_left_wrist_depth.mp4`（需 `camera_sender.enable_depth: true`）

如果要从一次长推理里切出每个任务：

- `task_segments.jsonl`
- `videos/frame_timestamps.jsonl`
- `trajectory_data/trajectory.parquet`

如果要确认脚踏板/任务热键标记的位置：

- `keyboard_markers.jsonl`
- `videos/frame_timestamps.jsonl`

如果要检查模型输入和输出：

- `eval_log.jsonl`
- `eval_images/`
- `eval_videos/`

### 视频时间戳

`videos/frame_timestamps.jsonl` 记录保存视频中每一帧对应的时间戳。它是一行一个 JSON，例如概念上类似：

```json
{
  "type": "video_frame_timestamp",
  "camera": "cam_mid_chest",
  "video_frame_index": 123,
  "image_timestamp": 1710000000.123,
  "state_timestamp": 1710000000.120,
  "inference_frame_index": 8,
  "action_step_index": 4,
  "record_timestamp": 1710000000.130
}
```

字段含义：

- `camera`：是哪一路相机，比如 `cam_left_wrist` 或 `cam_mid_chest`。
- `video_frame_index`：这一帧在 mp4 视频里的帧号，从 0 开始。
- `image_timestamp`：相机图像时间戳，用来和任务段、轨迹对齐。
- `state_timestamp`：这帧图像对齐到的机器人状态时间戳。
- `inference_frame_index`：对应第几次模型推理。
- `action_step_index`：对应当前 action chunk 里的第几步动作。
- `record_timestamp`：写入记录时的本机 wall time。

后处理时可以结合 `task_segments.jsonl` 或 `keyboard_markers.jsonl` 裁剪视频和轨迹。

### 任务段

`task_segments.jsonl` 只在 `task_mode: continuous_tasks` 下生成。它用来回答：同一个组目录里，第 1 个任务、第 2 个任务…分别从哪里开始、在哪里结束。

每个任务段有一条 `start` 和一条 `end`：

```json
{
  "type": "task_boundary",
  "event": "start",
  "task_segment_index": 1,
  "wall_time": 1234567890.0,
  "video_frame_counts": {"cam_mid": 123},
  "latest_camera_timestamps": {"cam_mid": 1234567890.0},
  "trajectory_rows": 0
}
```

字段含义：

- `event`：`start` 表示任务开始，`end` 表示任务结束。
- `task_segment_index`：第几个任务，从 1 开始。
- `wall_time` / `iso_time`：按键或任务边界发生时的本机时间。
- `video_frame_counts`：当时每路视频已经写入多少帧，可以直接用来定位视频切点。
- `latest_camera_timestamps`：当时最近一帧相机图像的时间戳。
- `trajectory_rows`：当时轨迹已经记录了多少行，可以用来裁剪 `trajectory.parquet`。

例如一次脚本运行采了 10 个任务，产物都在同一个 `action-step_10/`（或 `ood-language/`）里，`task_segments.jsonl` 会有 10 组 start/end。

### 人工 marker

`keyboard_markers.jsonl` 记录人工打点。推理时按配置的任务热键（默认 `Ctrl+.`），或者脚踏板映射为同一键，就会写一行：

```json
{
  "type": "keyboard_video_timestamp_marker",
  "marker_index": 1,
  "key": "ctrl+.",
  "source": "pynput",
  "wall_time": 1710000000.123,
  "video_frame_counts": {"cam_mid": 456},
  "latest_camera_timestamps": {"cam_mid": 1710000000.120},
  "trajectory_rows": 78,
  "inference_frame_index": 12,
  "task_segment_index": 2
}
```

它适合标记“这一刻发生了重要事件”，比如开始抓取、抓取成功、放置完成、人工觉得失败等。后处理时可以用 `video_frame_counts` 或 `latest_camera_timestamps` 找到视频中的位置。

### 轨迹

`trajectory_data/trajectory.parquet` 是后续做训练数据提取时最重要的文件。它保存每一步动作对应的状态、动作和时间戳。

主要字段：

- `timestamp`：这一行记录写入时的本机时间。
- `state_timestamp`：机器人状态时间戳。
- `image_timestamp`：对应图像时间戳。
- `inference_frame_index`：第几次模型推理。
- `action_step_index`：当前 action chunk 里的第几步。
- `task_segment_index`：属于第几个任务段；连续任务模式下用于拆分数据。
- `observation.state`：模型执行时对应的机器人状态向量。
- `action`：模型输出并执行的动作向量。

`task_segment_index` 可用于把同一个组目录内的多次任务拆成训练样本。

## 夹爪控制格式

推理端发送 3 个 `float64`：

```python
gripper_control = np.zeros(3, dtype=np.float64)
gripper_control[1] = action_full[14]
```

接收端使用：

```python
np.frombuffer(data, dtype=np.float64)
```

当前下位机 `lower_machine/only_gripper.py` 会读取：

- `flag`
- `left_cmd`
- `right_cmd`

并将 `left_cmd/right_cmd` clamp 到 `[0, 1]` 后由 `RoboTG_g.set_gripper_position` 写入 Dynamixel。

`flag=0` 和 `flag=2` 都会被接受；`flag=1` 表示停止。

## 常见问题

### 夹爪不动

先确认：

- 下位机 `only_gripper.py` 已运行
- 推理配置中 `network.low_host` 是夹爪控制机 IP。
- `network.gripper_port` 与接收端端口一致，默认 `30015`。
- 接收端终端能看到 `[GRIPPER_RECV]` 和 `[GRIPPER_EXEC]`。
- 下位机应看到 `[GRIPPER_STATE] 推送夹爪状态 -> 192.168.1.12:30016`。
- 上位机日志应有 `[GRIPPER_STATE_UDP] 监听夹爪状态` 与 `[DIRECT_GRIPPER] UDP 夹爪控制 + 状态回传监听端口 30016`。

### 推理里夹爪 state 不准

`gripper_mode: udp` 时，推理/数采 state 夹爪优先用下位机 Dynamixel 回传（`gripper_state_port`，默认 30016）。无包或超时则 state 夹爪不更新为指令值（数采会丢帧）。检查下位机 `only_gripper.py` 是否在跑、防火墙是否放行 UDP 30016。

### 相机窗口不显示

检查：

- 当前机器是否有显示器或可用 GUI。
- 是否通过 SSH 无显示环境运行。
- `session.visualize_cam_left` / `session.visualize_cam_mid` 是否为 `true`。

### 脚踏板没有写 marker

先运行：

```bash
python3 pedal_monitor.py
```

确认脚踏板实际是否发出配置的热键（如 `ctrl+.`）。推理时 marker 会写入：

```text
组目录/keyboard_markers.jsonl
```

### 想只跑一个连续 test，不自动分段

设置：

```yaml
session:
  task_mode: test
  max_duration_sec: null
```

此时一个组目录会一直运行，直到手动停止。

## 巡检命令

修改代码后可先跑：

```bash
python3 -m compileall -q infer data_collect common tools lower_machine pedal_monitor.py
python3 - <<'PY'
from infer.config_loader import load_infer_config
cfg = load_infer_config("infer/config/lerobot/pi05_left_arm_direct.yaml")
print(cfg.engine.value, cfg.session.task_mode, cfg.session.visualize_cam_mid)
PY
```
