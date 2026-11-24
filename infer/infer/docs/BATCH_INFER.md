# 批量真机推理 — 实习生操作手册

> 不用会写代码。按顺序做就行。  
> **全程只需要启动 1 次批量脚本**，后面全部在终端里按提示输入、用脚踏板操作。

---

## 这份流程是干什么的？

在一台真机上，自动完成一整套测试：

1. **阶段一**：测试 5 种不同的 action-step，找出哪个最好
2. **阶段二**：你在终端里选出「最好的那个 action-step」
3. **阶段三**：用最好的 action-step，测试 5 种不同的语言指令（OOD）

所有结果会保存在 **模型同级目录**里（与 `pretrained_model` 文件夹并列）。

例如目录结构：

```
.../5000/
├── pretrained_model/          ← 模型文件（在配置文件 policy.ckpt_dir 里改路径）
├── action-step_10/            ← 测试结果被写在这里
├── action-step_20/
├── ood-language/
└── batch_infer_summary.json
```

---

## 你需要提前确认的事

找老师确认下面这些信息，并抄到纸上：

| 项目 | 在哪里改 | 说明 |
|------|----------|------|
| 项目代码目录 | — | 终端里 `cd` 到这个目录 |
| **模型路径** | 配置文件 `policy.ckpt_dir` | 填到 `pretrained_model` 那一层 |
| **每组任务数 (action-step)** | `batch_infer.tasks_per_group` 或 `policy.tasks_per_group` | 阶段一每组独立任务数（如 20） |
| **OOD 任务段数** | `infer/config/ood_language_presets.yaml` 或 `ood_language_presets_dual_arm.yaml` | 阶段三整组固定 **5** 段，产物在 `ood-language/`；双臂配置用后者 |
| **每组初始位** | `direct_robot.reset_to_init_on_group_start` | 每组 Y 确认后回标准位姿（非每个任务） |
| 结果目录 | 不用改 | 自动写在 `pretrained_model` 同级 |
| 配置文件 | 见下方「选哪个配置文件」 | Pi0.5 或 ACT 各一份 direct 配置 |
| 脚踏板 | `pedal:` 段 | 默认 **Ctrl+.**=任务；**Y**=确认；**N**=取消；**Delete**=丢弃 |
| **WPS 评分表依赖** | 终端一次性安装 | `pip install -r infer/requirements-score-sheet.txt`（或 `pip install xlrd==1.2.0 xlutils xlwt`） |
| **WPS 表格软件** | 本机已安装 `et` / `wps` | 每组测试会自动弹出评分表 |
| **评分表开关** | `batch_infer.feedback_score_sheet` | `true`（默认）生成并打开 .et；`false` 跳过（调试/无 WPS 时） |
| **VLA 自动评估** | `batch_infer.vla_auto_eval` | `enabled: true`（默认）每组 action-step 结束写入 `eval_table/eval_result.et` + `summary.md` |
| **VLA 评估依赖** | 终端一次性安装 | `pip install -r vla_eval/requirements.txt`（含 PyYAML、xlrd 等） |

---

---

## 选哪个配置文件？

| 模型 / 场景 | 配置文件 | 启动命令 |
|------|----------|----------|
| **Pi0.5 左臂+左夹爪**（默认） | `infer/config/lerobot/pi05_left_arm_direct.yaml` | `python3 infer/cli/run_batch_infer.py` |
| **Pi0.5 右臂+夹爪** | `infer/config/lerobot/pi05_right_arm_direct.yaml` | `python3 infer/cli/run_batch_infer.py --config infer/config/lerobot/pi05_right_arm_direct.yaml` |
| **Pi0.5 双臂+左夹爪** | `infer/config/lerobot/pi05_dual_arm_gripper_direct.yaml` | `python3 infer/cli/run_batch_infer.py --config infer/config/lerobot/pi05_dual_arm_gripper_direct.yaml` |
| **ACT 左臂+左夹爪** | `infer/config/lerobot/act_left_arm_direct.yaml` | `python3 infer/cli/run_batch_infer.py --config infer/config/lerobot/act_left_arm_direct.yaml` |
| **原生 ALOHA ACT 左臂+左夹爪** | `infer/config/aloha/act_aloha_left_arm_direct.yaml` | `python3 infer/cli/run_batch_infer.py --config infer/config/aloha/act_aloha_left_arm_direct.yaml` |
| **原生 ALOHA DP 左臂** | `infer/config/aloha/dp_aloha_left_arm_direct.yaml` | `python3 infer/cli/run_batch_infer.py --config infer/config/aloha/dp_aloha_left_arm_direct.yaml` |
| **原生 ALOHA RDT 左臂** | `infer/config/aloha/rdt_aloha_left_arm_direct.yaml` | `python3 infer/cli/run_batch_infer.py --config infer/config/aloha/rdt_aloha_left_arm_direct.yaml` |
| **原生 ALOHA ACT/DP/RDT 双臂** | `infer/config/aloha/*_dual_arm_gripper_direct.yaml` | 同上，换对应 `--config` |
| **DP 左臂+左夹爪** | `infer/config/lerobot/dp_left_arm_direct.yaml` | `python3 infer/cli/run_batch_infer.py --config infer/config/lerobot/dp_left_arm_direct.yaml` |

Pi0.5 各配置的 `policy.keep_dims` 须与 checkpoint 维数一致：左臂 8 维、右臂 8 维、双臂 16 维。双臂配置 OOD 阶段使用 `batch_infer.ood_presets: ../ood_language_presets_dual_arm.yaml`（相对实验 YAML；文件在 `infer/config/`）。

LeRobot ACT / DP 额外必填：`policy.dataset_repo_id`、`policy.dataset_root`（须指向含 `meta/stats.json` 的数据集目录）。原生 ALOHA：ACT 用 `act`+`act_backend: aloha`；DP 用 `type: aloha_dp`；RDT 用 `type: aloha_rdt`；均读 checkpoint 同目录 `dataset_stats.pkl`。若 `init_pose_source: dataset_median`，仍需配置 `dataset_root`，否则会回退 standard 初始位。

ACT / DP **只有 action-step 阶段**（无 OOD）；配置 `batch_infer.phase: action-step`。启动时可**自定义 action-step**，完成后可追加其他 step，**不会删除已有 action-step 目录**（仅「全部从头」会清空）。

组间初始位：ACT/DP 配置可设 `direct_robot.init_pose_source: dataset_median`，从训练集各 episode **首帧**取中位数后再开始本组推理（左臂+夹爪来自数据集，腰/右臂仍用 `standard_init_joint_degrees` 补全）。

> ACT / DP **不吃语言 prompt**；若只做 action-step 消融，保持 `batch_infer.phase: action-step` 即可。

---

## 老师让你改配置时（改 yaml，不用改启动命令）

打开对应 **direct** 配置文件，常用项（**只改这一个文件**）：

- Pi0.5 左臂：`infer/config/lerobot/pi05_left_arm_direct.yaml`
- Pi0.5 右臂：`infer/config/lerobot/pi05_right_arm_direct.yaml`
- Pi0.5 双臂+左夹爪：`infer/config/lerobot/pi05_dual_arm_gripper_direct.yaml`
- ACT：`infer/config/lerobot/act_left_arm_direct.yaml`
- 原生 ALOHA ACT：`infer/config/aloha/act_aloha_left_arm_direct.yaml`
- 原生 ALOHA DP：`infer/config/aloha/dp_aloha_left_arm_direct.yaml`
- 原生 ALOHA RDT：`infer/config/aloha/rdt_aloha_left_arm_direct.yaml`
- DP：`infer/config/lerobot/dp_left_arm_direct.yaml`

```yaml
policy:
  ckpt_dir: /home/.../5000/pretrained_model   # 模型路径（改这里）
  tasks_per_group: 25                         # action-step 每组任务数
  # ACT / DP 还需:
  # dataset_repo_id: your_repo
  # dataset_root: /path/to/dataset   # 须含 meta/stats.json

batch_infer:
  ood_tasks_per_group: 1                      # 已废弃；OOD 段数由 presets 数量决定（通常 5）
  action_step_parts: 5
  phase: all
  # action-step 每组结束后自动评估（与 WPS 评分表独立，可单独开关）
  vla_auto_eval:
    enabled: true
    output_dirname: eval_table
    result_table_name: eval_result.et
    summary_name: summary.md

pedal:                                        # 脚踏板按键（改这里，不用改代码）
  task_toggle: ctrl+.                         # 开始/结束每个独立任务
  confirm: y
  cancel: n
  discard: delete
```

> 启动时会打印 `任务/组 (action-step)` 与 `任务/组 (OOD)`，用于核对配置。

### task_prompt 改哪里？

| 阶段 | 生效配置 | 说明 |
|------|----------|------|
| **action-step** | `policy.task_prompt`（direct yaml 第 48 行附近） | 改这里即可；**不会**被 `infer/config/ood_language_presets.yaml` 覆盖 |
| **OOD** | `infer/config/ood_language_presets.yaml` 各条 `task_prompt` | 每段自动切换；`baseline_prompt` 仅作 OOD 菜单展示与元数据 |

改完 prompt 后若要更新已有 `action-step_xx/` 目录里的日志，需**重做该组**（续跑时选「清空重做」或删除该目录），否则 `meta.json` / `eval_log.jsonl` 仍是旧 run 的内容。

> ### 🔴 实习生日常只跑命令，不要自己乱改配置；要改找老师。

---

### 🔴 安全（必看）

> **机器人旁边必须有人看着。**  
> **出现异常立刻按 Ctrl+C 停程序，必要时按机器人急停。**  
> **不要在机器人运动时伸手进工作空间。**

---

## 一共要开几个终端？

### ⚠️ 需要 2 个终端窗口

| 终端 | 干什么 | 能不能关 |
|------|--------|----------|
| **终端 A** | 夹爪控制（先开，一直开着） | ❌ 全程不能关 |
| **终端 B** | 批量推理主程序（只启动 1 次） | 全部做完或 Ctrl+C 后才能关 |

---

## 第一步：打开终端 A — 启动下位机（夹爪）

1. 在**下位机** (192.168.1.13) 打开终端。
2. **逐行**复制粘贴（路径改成实际项目目录）：

```bash
cd /home/你的用户名/infer/lower_machine
python3 run.py --gripper-only
```

> exo_safety 启用时用 `python3 run.py`（不加 `--gripper-only`）。

---

## 第二步：打开终端 B — 启动批量推理（只执行 1 次）

1. **再开一个新的终端**（不要关掉终端 A）。
2. **Pi0.5 左臂（默认）只需下面三行**（路径已在默认配置文件里）：

```bash
cd /home/你的用户名/infer
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 infer/cli/run_batch_infer.py
```

**Pi0.5 右臂 / 双臂 / ACT** 需加 `--config`，例如：

```bash
# 右臂
python3 infer/cli/run_batch_infer.py --config infer/config/lerobot/pi05_right_arm_direct.yaml

# 双臂 + 左夹爪
python3 infer/cli/run_batch_infer.py --config infer/config/lerobot/pi05_dual_arm_gripper_direct.yaml

# ACT
python3 infer/cli/run_batch_infer.py --config infer/config/lerobot/act_left_arm_direct.yaml
```

> ### 🔴 左臂 Pi0.5 不要加多余参数；其他场景必须指定 `--config`。
>
> 模型路径、每组任务数等都在对应的 **`*_direct.yaml`** 里，老师改配置文件即可。

3. 启动后终端会显示类似：

```
批量推理启动
  配置文件: .../pi05_left_arm_direct.yaml
  模型目录: .../5000/pretrained_model
  结果目录: .../5000
  chunk_size: 50
  action-step 候选: [10, 20, 30, 40, 50]
  任务/组 (action-step): 20
  任务/组 (OOD): 5
  模型/硬件: 已加载并连接，全程保持不断开
```

> ### 🔴 启动后会有约 2 秒「缓冲区预填充」，只有这一次，请等着，不要按键。
>
> 之后换组测试时**不会**再断开机器人/相机，也**不会**再等 2 秒。

---

## 脚踏板怎么用？（整个流程最重要）

批量推理使用 **4 个脚踏板**（或映射为对应键盘键）：

| 脚踏板 | 键盘键 | 作用 |
|--------|--------|------|
| 任务 | **Ctrl+.** | 开始 / 结束每个独立任务 |
| 确认 | **Y** | 确认开始本组测试（原终端 y） |
| 取消 | **N** | 跳过本组 / 不继续（原终端 n） |
| 丢弃 | **Delete** | 丢弃当前任务数据，方便误操作后重做 |

> 按键在配置文件 `pedal:` 段设置（默认 `task_toggle: ctrl+.`）。脚踏板软件里映射成对应键即可，**改 yaml 不用改代码**。若要换回 `Ctrl+G`，把 `task_toggle` 改成 `ctrl+g`。

### 任务脚踏板（Ctrl+.）

每一「个独立任务」的操作：

```
第 1 次 Ctrl+.  →  开始任务（机器人开始动）
第 2 次 Ctrl+.  →  结束任务（这一段算完成）
第 3 次 Ctrl+.  →  开始下一个任务
第 4 次 Ctrl+.  →  结束下一个任务
…… 以此类推
```

### 丢弃脚踏板（Delete）

误操作时可按 **Delete** 撤销数据（不用关程序）：

- **任务进行中**：清除本段已录制的轨迹/帧，在本段内重新做（任务编号不变）
- **刚结束上一段、等待下一次 Ctrl+.**：撤销上一段，按 Ctrl+. 重做该任务

终端会打印 `[PEDAL] …` 提示是否成功（键名与 `pedal.discard` 配置一致）。

### 确认 / 取消脚踏板（Y / N）

原先终端问 `(y/n)` 的地方（如「确认开始本组？」「是否继续 OOD？」）改为：

- 终端显示提示 + `[PEDAL] Y=确认  N=取消`
- 按 **Y** 或 **N** 脚踏板即可，无需在键盘上输入 y/n

> ### 🔴 action-step 每组要做满 N 个独立任务（默认 N=20，见 `policy.tasks_per_group`）
>
> OOD 阶段整组 **5** 个独立任务段（对应 5 种语言指令），串行录在 **`ood-language/`** 组目录里；每段按 **2 次** Ctrl+.（开始+结束），共 **10 次** Ctrl+. 完成整组。
>
> action-step 一组 = N 个任务 = **2×N 次** Ctrl+.；做完第 N 个任务并结束 Ctrl+. 后，程序自动结束该组。

### 做任务时注意

- 等终端提示 `等待 Ctrl+. 开始第 1 个任务段` 再按第一次 Ctrl+.。
- 任务进行中不要乱按 Ctrl+.，除非你要**提前结束**这一段。
- 一组里相机录的是 **1 条连续视频**；每个任务的开始/结束时间会记在日志里，不用你手动记。
- **Delete 只清轨迹/相机帧缓存**，`eval_log.jsonl` 里已写的推理记录不会自动删（一般不影响主流程）。

---

## 阶段一：Action-step 测试（共 5 组）

程序会**按顺序**问你测 5 个 action-step，例如：`10、20、30、40、50`。

每一组流程：

```
终端提示「确认开始 [action-step] …？」+ [PEDAL] Y=确认 N=取消
    ↓ 按 Y 脚踏板
机械臂自动回到标准初始位（每组 1 次，与 new_exo_control_mode_lock 一致）
    ↓ 等待约 1.5 秒
按 Ctrl+. 开始第 1 个任务
    ↓ 做任务（误操作可按 Delete 丢弃本段重做）
按 Ctrl+. 结束第 1 个任务
    ↓ 重复
…… 共 20 个任务 ……
    ↓ 第 20 个任务结束
这一组自动结束
    ↓
下一组 action-step，按 Y 继续
```

> ### 🔴 5 组全部测完之前，不要关终端 B，不要关终端 A。

每组用的语言指令是**默认的**（不用你改）。

结果保存在（`pretrained_model` 的同级目录）：

```
.../5000/
  action-step_10/
  action-step_20/
  action-step_30/
  action-step_40/
  action-step_50/
```

---

## 阶段二：选出最好的 action-step

5 组都跑完后，终端会问：

```
请输入最优 action-step (10, 20, 30, 40, 50):
```

> ### 🔴 看机器人实际表现，输入你觉得成功率最高的那个数字，回车。
>
> 不确定就问老师一起看录像再选。

---

## 阶段三：OOD Language 测试（语言指令测试）

固定用你刚选好的 action-step，**一次整组**测完 5 种语言指令（与 action-step 一组 20 个任务类似）。

终端会先列出任务段 1～5 与对应 ID：

| 段 | ID | 大概测什么 |
|----|-----|------------|
| 1 | 1 | 把「左夹爪」改成「右夹爪」 |
| 2 | 2 | 把物体名 toy 改成 object |
| 3 | 3 | 把黄色盒子改成蓝色 |
| 4 | 4 | 完全无关的句子 |
| 5 | 5 | 轻微改几个字 |

流程：

```
终端：确认开始 [ood-language] 整组 5 个 OOD 任务段？ + [PEDAL] Y=确认
    ↓ 按 Y，WPS 自动打开 OOD 评分表
按顺序做 5 个独立任务段（每段 Ctrl+. 开始 + Ctrl+. 结束；程序自动切换 task_prompt）
    ↓ 5 段全部结束
终端：请在 WPS 保存并关闭评分表 → 按 Y
    ↓
OOD 阶段结束
```

> ### 要点
>
> - **1 个组目录**：5 段视频/日志串行录在 `ood-language/` 下，对应评分表 test 1～5 列。  
> - **顺序固定**：按上表 1→5 执行，每段开始前程序自动切换提示词。  
> - **重测整组**：续跑菜单选重做，或删 `ood-language/` 后重跑。

结果保存在：

```
.../5000/
  ood-language/                  ← 整组 5 段（产物直接在此目录）
    videos/
    task_segments.jsonl          ← 5 条任务段记录
    机械臂抓取模型反馈评分ood.et
    ood_group_meta.json          ← 各段 ID / prompt 映射
```

---

## 全部做完之后

此时若老师已开启远程同步，程序会再问一次：

```
[REMOTE] 实验结果将上传到: user@host:/path/...
  约 1.2 GB（156 个文件）
是否开始上传？
  [PEDAL] Y=确认  N=取消
```

**你只需要按 Y 开始上传，或按 N 跳过。** 不需要核对文件列表，也没有其他确认步骤。上传结束后程序才退出。

若未配置远程同步，会直接显示 `[OK] 批量推理完成`，两个终端都可以关了。

### 交给老师检查的文件

在 **结果目录**（`pretrained_model` 同级，例如 `.../5000/`）下应有：

```
.../5000/
├── batch_infer_summary.json    ← 总摘要
├── batch_infer_log.jsonl       ← 每组运行记录 + 终端交互（phase=interaction）
├── action-step_10/             ← 阶段一各组（产物直接在此目录）
├── action-step_20/
├── …
├── pretrained_model/           ← 模型文件（不要删）
└── ood-language/               ← 阶段三整组（5 个任务段）
```

每个组目录（如 `action-step_10/`、`ood-language/`）里应有：

- `videos/` — 相机视频（每路相机 **1 个** mp4）
- `eval_table/` — **VLA 自动评估**（仅 action-step 阶段；`batch_infer.vla_auto_eval.enabled=true` 时生成）
  - `eval_result.et` — 自动填写的评测表（模板默认 `vla_eval/template/评测表.et`，按分值填写）
  - `summary.md` — 成功率与失败原因统计
- `task_segments.jsonl` — 各独立任务的开始/结束时间戳（action-step 约 N 条，OOD 为 5 条）
- `run.log` — 人类可读日志
- `inference_summary.json` — 本组结束摘要
- **评分表**（程序自动生成并预填配置，人工填写记录后保存；模板位于 `infer/template/`）：
  - action-step 阶段：`infer/template/机械臂抓取模型反馈评分模型.et`
  - OOD 阶段：`infer/template/机械臂抓取模型反馈评分ood.et`

> 每组确认开始（Y）后，程序会从模板生成评分表并**自动用 WPS 打开**；配置项（action-step、推理步数、提示词、video/模型路径等）已写好，你只需在表格里填测试记录。本组全部任务结束后，终端会提示 **保存并关闭表格**，保存后按 **Y** 继续。
>
> 若不需要评分表（调试、无 WPS、未装 xlrd 等），在配置文件里设 `batch_infer.feedback_score_sheet: false`。
>
> 若不需要 VLA 自动评估，设 `batch_infer.vla_auto_eval.enabled: false`。

### VLA 评估与主推理的隔离

| 项 | 行为 |
|----|------|
| **触发时机** | 仅在 `run_persistent_group()` 整组推理结束后执行 |
| **OOD 阶段** | `phase_label != action-step` 时直接跳过 |
| **读写范围** | 只读该组日志；只写 `eval_table/`，不改 `eval_log.jsonl` 等推理产物 |
| **依赖缺失** | `vla_eval` / PyYAML / xlrd 未安装 → 打 WARN 跳过，**不**在 `BatchRunner.__init__` 校验 |
| **评估异常** | 模板缺失、写盘失败、规则错误等 → 打 WARN，继续下一组 |
| **与 WPS 评分表** | 人工评分表保存确认先于 VLA 评估，评估耗时不阻塞 WPS 流程 |

---

## 常见问题

### 问：中途想停怎么办？

在**终端 B** 按 **Ctrl+C**。机器人/相机会自动断开。  
下次再跑**同一条启动命令**即可。程序会先检测该模型是否已有结果，并询问如何续跑（见下方「断点续跑」）。

### 问：程序意外中断后，怎么接着测、不从头来？

启动时会自动扫描 **结果目录**（`pretrained_model` 同级，如 `.../5000/`），根据 `batch_infer_summary.json`、各 `action-step_*` / `ood-language/` 目录判断进度。

**从未测过** → 直接开始新测试。

**测到一半** → 终端会显示每个 action-step 的状态（已完成 / 组内中断 / 未开始），例如：

```
[partial] action-step_40: 中断于 action-step_40 （已完成 9/20 个独立任务，续跑将整组从任务 1 重测）
[not_started] action-step_50: 未开始
```

然后让你选：

```
  [1] 从中断处继续（跳过已完成的 action-step；当前项组内从独立任务 1 整组重测）
  [2] 只重做当前项/阶段
  [3] 全部从头开始（清空 action-step 与 ood-language）
  [4] 退出
```

选 **[3] 全部从头** 时，还会进入 **二次确认** 页面，列出将被清空的内容；选 **[2] 返回上一步** 可回到主菜单，**不会删除任何文件**。

```
  [1] 确认：清空上述结果并从头重测
  [2] 返回上一步（不删除任何文件）
```

确认后才会：各 `action-step_*` 下的 test 清空重测；`ood-language/` 下 test 与 meta 删除；summary 里 `best_action_step` / `ood_results` 清除。

**交互日志**：上述菜单选择、二次确认、选最优 step、OOD ID、组开始/跳过、远程上传确认等，都会写入 `batch_infer_log.jsonl`（`phase: interaction`，含 `event` 字段），控制台也会打 `[BATCH][INTERACTION]`。

**重要**：一组里视频和日志是 **串行** 录在同一个组目录（如 `action-step_40/`）里的。  
若在 action-step_40 的第 10 个任务处中断，续跑 **不是** 从第 10 个任务接着录，而是 **清理该组半成品，从第 1 个独立任务整组重测 action-step_40**；10/20/30 若已完整完成则跳过。

**选择最优 action-step 与 OOD 属于同一阶段**：action-step 全部测完后，必须先**在终端输入**成功率最高的 action-step，才会进入 OOD。若 action-step 刚测完就中断、尚未输入最优 step，续跑 **[1]** 会先让你选择最优 step，**不会**直接进 OOD。只有终端确认后（`batch_infer_summary.json` 里 `best_action_step_confirmed: true`）才会记为「已选最优」；若重测了 action-step，需重新选择。

**全部测完** → 菜单示例：

```
  [1] 退出，不重新测试
  [2] 上传实验结果到服务器（上传失败 / 尚未上传 / 上次跳过）
  [3] 全部从头开始（清空 action-step 与 ood-language）
```

**只有全部任务都已完成** 时才会出现 **[2] 上传**；测到一半不会出现上传选项。上传已成功则菜单只有 [1] 退出、[2] 全部从头。

> 测到一半时一般选 **[1] 从中断处继续** 即可。

### 问：某一组做坏了想重跑？

- **阶段一**：跟老师说，可能需要删对应目录（如 `action-step_20/`）后重跑。
- **阶段三**：续跑菜单选「重做 OOD 整组」，或删 `ood-language/` 后重跑。

### 问：终端一直显示「等待 Ctrl+.」不动？

正常。说明在等你按脚踏板（Ctrl+.）**开始**任务。

### 问：按了 Ctrl+. 没反应？

- 确认光标在**终端 B** 所在窗口（有时焦点不在终端上按键无效）。
- 脚踏板是否映射成 Ctrl+.，问老师确认。
- pynput 全局监听一般不需要焦点；若完全无反应，看终端是否有「脚踏板监听已启动」。

### 问：Delete 脚踏板没反应？

- 确认映射为键盘 **Delete**（不是 Backspace）；若仍无效，在配置里改 `pedal.discard`（如 `backspace` 或单字符键）。
- 仅在「任务进行中」或「刚结束上一段、等待下一次任务键」时有效。
- 终端应出现 `[PEDAL] …` 日志；若完全没有，检查是否安装了 `pynput`。
- **Delete 只清内存中的轨迹/相机帧缓存**；`eval_log.jsonl` 已写入的记录不会自动删除（一般不影响主流程）。

### 问：报错「模型目录不存在」？

找老师检查配置文件里的 **`policy.ckpt_dir`** 是否填对（要填到 `pretrained_model`）。

---

## 远程同步（老师配置，实习生只看 Y/N）

全部阶段正常结束后，程序自动打包实验结果并询问是否上传。**实习生唯一操作：按 Y 上传，或按 N 跳过。**

上传内容由程序自动处理（跳过模型权重、超过 1GB 的单文件），`training_state/` 内 `.json` 等小文件仍会上传。

老师可在 yaml 中配置：

```
ckpt_dir: .../0423-0416-leftonly-c50-a50-vla/5000/pretrained_model
上传源:   .../5000/   （pretrained_model 同级，仅当前 step 的消融结果）
远程目标: {remote_dir}/0423-0416-leftonly-c50-a50-vla/5000/
```

若改用 `2000` 模型，则只同步 `2000/` 目录，远程为 `{remote_dir}/0423-0416-leftonly-c50-a50-vla/2000/`。

在对应 **direct** 配置文件（如 `pi05_left_arm_direct.yaml`、`pi05_dual_arm_gripper_direct.yaml`）中配置：

```yaml
batch_infer:
  remote_sync:
    enabled: true
    host: "192.168.1.100"
    port: 22
    username: "your_user"
    remote_dir: "/data/infer_results"
    private_key_path: ~/.ssh/id_rsa
```

**老师须知**（实习生可跳过本节）：

- 上传源为 **`policy.ckpt_dir` 的父目录**（`pretrained_model` 同级，即 `.../5000/`），**只传当前 step**
- 远程目录为 **`{remote_dir}/{模型run名}/{step}`**（如 `0423-0416-leftonly-c50-a50-vla/5000`），用模型 run 名区分不同实验，避免多个模型共用 `2000`/`5000` 时互相覆盖
- `pretrained_model/` 只传 `.json`，权重自动跳过
- 任意路径下单文件 **> 1 GiB** 自动跳过（如 `training_state/optimizer_state.safetensors`）；同目录下 `.json` 等小文件仍会上传
- **详细审计日志**（给老师，实习生终端看不到）：
  - `batch_infer_log.jsonl`：`phase=remote_sync`，含 `plan`（每个 upload/skip 文件）、`start`/`complete`/`declined`/`error`（含 rsync 输出、SFTP 逐文件记录）
  - `remote_sync_manifest.jsonl`：逐文件清单
  - `remote_sync_confirm` 仍在 `phase=interaction`（仅 Y/N 结果）
- SSH 私钥默认 `~/.ssh/id_rsa`；优先 `rsync`，失败回退 `paramiko`
- 同步失败：终端只显示「上传失败，请联系老师」；具体原因在日志里

---

## 命令速查（复制用）

**终端 A（夹爪，先开，一直开着）：**

```bash
cd /home/你的用户名/infer
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 lower_machine/run.py --gripper-only
```

**终端 B（批量推理，只开 1 次）：**

左臂 Pi0.5（默认，无需 `--config`）：

```bash
cd /home/你的用户名/infer
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 infer/cli/run_batch_infer.py
```

双臂 Pi0.5 示例：

```bash
python3 infer/cli/run_batch_infer.py --config infer/config/lerobot/pi05_dual_arm_gripper_direct.yaml
```

---

## 操作流程一张图

```
[终端 A] 夹爪程序 ────────────────────────────── 全程开着
                │
[终端 B] 批量脚本（只启动 1 次）
                │
                ├─ 阶段一：action-step 10 → 20 → 30 → 40 → 50（每组 20 任务）
                │
                ├─ 阶段二：终端输入最优 action-step
                │
                └─ 阶段三：OOD 整组 5 段，产物在 `ood-language/`，做完保存评分表
                │
                ▼
         结果在 pretrained_model 同级目录下（如 .../5000/）
```

---

**有问题先喊老师，不要自己乱改配置或代码。**
