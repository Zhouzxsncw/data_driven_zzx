# PI0.5 推理部署总览（TongBot 真机）

> 2026-08-26 新增。本文是 PI0.5 推理的**顶层入口**：怎么选后端、怎么换新模型、
> 换模型 checklist、与 LingBot-VLA V2 流程的差异。模块细节见文末导航表。
> LingBot-VLA V2 的部署总览另见 `~/CIGAI/README_推理部署流程.md`。

---

## 一、架构一句话

PI0.5 是**单进程**推理：`run_batch_infer.py` 一个进程内直接加载模型 → 相机/机械臂 → 出动作，
**不需要**像 LingBot-VLA V2 那样开 WebSocket 服务端双终端。

```
终端 1（唯一）:
  conda activate openpi-pi05-infer
  cd ~/CQTongBot/scripts   # PYTHONPATH 用仓库根
  python infer/run_batch_infer.py --config infer/config/lerobot/<选一个 pi05 yaml>
      └─ 进程内: 加载 ckpt(含 norm) → RealSense 3 相机 → UDP 机械臂 → 动作
```

---

## 二、选哪种后端？（看 ckpt 格式，不看心情）

| ckpt 里有什么 | 用哪套配置 | `pi05_backend` | 环境 | 是否需要 JAX |
|---|---|---|---|---|
| `pretrained_model/config.json`（LeRobot 训练产物） | `pi05_*_direct.yaml` | `lerobot`（默认） | openpi-pi05-infer 亦可 | 否 |
| `params/` 或 `model.safetensors`（OpenPI 原生） | `pi05_openpi_*.yaml` | `openpi` | `openpi-pi05-infer` | **是** |

**判别方法**：进 ckpt 目录看一眼。
- 有 `config.json` + `model.safetensors` + `preprocessor_config.json` → LeRobot 格式
- 有 `params/`（orbax 目录）或 `assets/` → OpenPI 格式

**两套不要混用**：LeRobot 权重喂 openpi 后端（或反过来）会直接报错或维数爆炸。

---

## 三、换新 ckpt 完整流程

### 线 A：LeRobot 后端（最简单，norm 内嵌 ckpt）

```
1. 训练机导出 pretrained_model/ 整目录
2. 传到推理机任意位置（如 ~/CIGAI/<实验名>/checkpoints/<step>/pretrained_model）
3. 改 YAML（只改一处）:
   policy.ckpt_dir: <上面的路径>/pretrained_model     ← 指到 pretrained_model 这层
4. 确认 keep_dims 与训练维数一致:
   左臂 ckpt=8 维、双臂=16 维（keep_dims 长度必须相等）
5. 启动（见第一节命令）
```

norm_stats 内嵌在 LeRobot ckpt 里自动读，**无同步问题**。

### 线 B：OpenPI 后端（norm 有回退链，要盯日志）

```
1. 训练机导出 ckpt（含 params/ 或 model.safetensors，最好带 assets/norm_stats.json）
2. 传到推理机
3. 改 YAML:
   policy.ckpt_dir:            <ckpt 目录>
   openpi_config_name:         与训练一致（如 pi05_aloha）
   openpi_action_dim:          16（双臂）/ 8（单臂）
   openpi_delta_action_dims:   必须与训练时一致（TongBot 双臂 [7,7,-1,-1]）
   openpi_camera_names:        与训练相机名一致
   task_prompt:                新任务指令
4. 启动后盯日志，必须看到:
   "使用 checkpoint assets 归一化统计: ..."      ← ✅ 正确
   若看到 "生成本地恒等统计" (identity)          ← ❌ 立即停机！见第五节
```

**OpenPI norm 加载顺序**（找不到会逐级回退，最后一级是坑）：
```
ckpt_dir/assets/norm_stats.json → TrainConfig assets_dir → 本机缓存 → identity(不归一化!)
```
训练用了真实 quantile 归一化 + 推理落到 identity = **全身乱动**。
预防：传 ckpt 时把训练端的 `norm_stats.json` 放进 `ckpt_dir/assets/` 一起带走。

---

## 四、换新数据集 checklist（PI05 版）

| 检查项 | 在哪 | 说明 |
|---|---|---|
| keep_dims 维数 | infer YAML | 左臂 8 / 右臂 8 / 双臂 16；数采原生 `[左7\|右7\|左爪\|右爪]` = 恒等 [0..15] |
| norm_stats | ckpt 内 | LeRobot 自动内嵌；OpenPI 确认 assets/ 带上（见上） |
| delta mask | `openpi_delta_action_dims` | 必须与**训练配置**一致，不能只在推理端硬套 |
| 相机名 | `openpi_camera_names` / `camera_keys` | 与训练数据集相机 key 一致 |
| 训练配置名 | `openpi_config_name` | 与训练一致；TongBot 16D 沿用 pi05_aloha + YAML 覆盖 |
| J7 数据修复状态 | 数据集本身 | 旧数据集（数采 J7 −120° 修复前）× delta = J7 转圈，须重采重训 |

---

## 五、历史事故速查（症状 → 病因）

| 症状 | 病因 | 详见 |
|---|---|---|
| 全身乱动、夹爪 ±1 | norm 落到 identity 回退 | `policies/openpi_pi05_runtime.md` §10.3 |
| 仅 L7/R7 匀速转圈（Δ≈−2.09 rad） | 数采 J7 坐标系 × delta 监督 | `docs/REPORT_PI05_J7_SPIN_AND_INFER_ISSUES.md` |
| 维数报错 `(16,) (14,)` | 14D Aloha stats 套 16D / keep_dims 错 | runtime 会拒收，查启动日志 |
| 加载就崩 | LeRobot/OpenPI 后端与 ckpt 格式混用 | 本文第二节 |

---

## 六、与 LingBot-VLA V2 流程对照

| | PI0.5（本文） | LingBot-VLA V2 |
|---|---|---|
| 架构 | 单进程，客户端直载模型 | 双终端：WebSocket 服务端(8006) + 客户端 |
| norm_stats | LeRobot 内嵌 ckpt / OpenPI assets/ 跟 ckpt 走 | 打包脚本随 checkpoint 包走（2026-08-26 起） |
| 换模型改几处 | 1 处（ckpt_dir），OpenPI 加 delta/相机名 | --model_path + lingbot_robo_name（新 pipeline 也是 2 处） |
| 语言 prompt | YAML `task_prompt` | 同左 |
| 环境名 | openpi-pi05-infer（含 JAX） | lingbotvla（服务端）/ openpi-pi05-infer（客户端） |

---

## 七、文档导航（按需深入）

| 文档 | 内容 |
|---|---|
| [`policies/pi05_adapter.md`](../policies/pi05_adapter.md) | Pi0.5 Adapter 流程（模型特点/RTC/chunk/J7） |
| [`policies/openpi_pi05_runtime.md`](../policies/openpi_pi05_runtime.md) | OpenPI runtime 详解（norm 回退链/delta/16D/J7 全 12 节） |
| [`docs/OPENPI_ENV.md`](OPENPI_ENV.md) | OpenPI 环境（vendored 源码/依赖/启动/agilex 适配边界） |
| [`docs/BATCH_INFER.md`](BATCH_INFER.md) | 批量消融实习生手册（action-step + OOD 全流程） |
| [`docs/REPORT_PI05_J7_SPIN_AND_INFER_ISSUES.md`](REPORT_PI05_J7_SPIN_AND_INFER_ISSUES.md) | J7 转圈 + norm 事故复盘 |
| [`config/README.md`](../config/README.md) | 全部实验配置索引 |
| `~/CIGAI/README_推理部署流程.md` | LingBot-VLA V2 部署总览（对照用） |

## 八、常用命令备忘

```bash
# 环境 + PYTHONPATH
conda activate openpi-pi05-infer
cd ~/CQTongBot/scripts && export PYTHONPATH="$(pwd):$PYTHONPATH"

# 双臂（OpenPI 后端）
python infer/run_batch_infer.py --config infer/config/lerobot/pi05_openpi_dual_arm_gripper_direct.yaml

# 双臂（LeRobot 后端）
python infer/run_batch_infer.py --config infer/config/lerobot/pi05_dual_arm_gripper_direct.yaml

# 单测（无 GPU 也能跑）
python3 -m pytest infer/tests/test_openpi_pi05_runtime.py -v
```
