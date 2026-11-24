# Pi0.5 推理流程

对应代码：`pi05_adapter.py`  
示例配置：`infer/config/lerobot/pi05_left_arm_direct.yaml`  
注册类型：`policy.type: pi05`

---

## 模型特点

- **VLA**（Vision-Language-Action）：图像 + 关节状态 + **语言 prompt**
- 可选 **RTC**（Real-Time Chunking）：推理延迟大于控制周期时，用 chunk 队列填补空窗
- 可选 **action chunk** 开环：`use_action_chunk: true` 时一次预测整段动作
- 归一化 stats **内嵌于 checkpoint**，无需单独 `dataset_root`

---

## 真机主链路（与其他模型共用）

```
infer/cli/run_infer.py
  → InferSession.run()
    → native_runner.run_native_session()
      → InferEngine.run()
```

单帧循环：

```
1. SyncEngine.align_at_wall_time()        # 相机 + 机械臂状态时间对齐
2. unpack 15D state + 多路 RGB
3. build_frame(images, state, task_prompt) # Pi0.5 独有：写入 task 语言字段
4. run_timed_inference(frame)
5. 执行 min(chunk_len, n_action_steps) 步 → expand_action_to_15d → UDP 发臂
6. 录制 machine_flow / eval_log / 视频
```

---

## Adapter 内部流程

### load()

```
PI05Policy.from_pretrained(ckpt_dir)
  → make_pre_post_processors（归一化来自 checkpoint）
  → 应用 n_action_steps_override / num_inference_steps_override
  → 若 rtc_enabled：
       resolve RTC 配置（合并 checkpoint + YAML）
       初始化 ActionQueue + LatencyTracker
```

### build_frame(images, state_15d, prompt)

| 字段 | 内容 |
|------|------|
| `observation.cam_left` | HWC → [C,H,W] float [0,1] |
| `observation.cam_mid` | 同上 |
| `observation.state` | `state_15d[keep_dims]` |
| `task` | `prompt` 或 `cfg.task_prompt` |

> 不走 `tools/policy/policy_frame_utils`（Pi0.5 preprocessor 对 batch 维约定不同）。

### run_timed_inference(frame)

```
preprocess(frame) → obs
  → _policy_action_step(obs)
       ├─ [无 RTC]
       │    ├─ use_action_chunk ? predict_action_chunk : select_action
       │    └─ postprocess
       └─ [有 RTC]
            ├─ 队列非空 → pop 一步
            └─ 队列空 → 带 prev_chunk_left_over + inference_delay 推理新 chunk
                         → merge 入队 → pop 一步
  → TimedActionChunk(post_action, rtc_info, 各阶段耗时)
```

---

## YAML 配置要点

| 字段 | 说明 |
|------|------|
| `type: pi05` | 注册到 `Pi05Adapter` |
| `pi05_backend` | `lerobot`（默认）或 `openpi`（进程内，源码在 `infer/policies/openpi`） |
| `ckpt_dir` | LeRobot：`pretrained_model`；openpi：含 `params/` 或 `model.safetensors` 的目录 |
| `task_prompt` | 默认语言指令 |
| `use_action_chunk` | `true`：一次预测 chunk；`false`：逐步 select_action（仅 LeRobot） |
| `rtc_enabled` | 是否启用实时 chunk 队列（**仅 LeRobot**；openpi 暂不支持） |
| `rtc_config` | RTC 参数覆盖（execution_horizon、max_guidance_weight 等） |
| `n_action_steps_override` | 开环步数上限（≤ 训练 chunk_size / openpi_action_horizon） |
| `num_inference_steps_override` | 扩散去噪步数覆盖（LeRobot） |
| `keep_dims` | 模型动作对应的状态维度 |
| `openpi_repo_path` | **已废弃**（可选覆盖）；默认用内置 [`infer/policies/openpi`](openpi/) |
| `openpi_config_name` | 如 `pi05_aloha` |
| `openpi_camera_names` | `[left, front, right]` → left_wrist / base / right_wrist |
| `openpi_action_dim` | 默认 14（AlohaOutputs）；须与 `keep_dims` 长度一致 |
| `openpi_action_horizon` | 默认 50；batch 消融上限 |
| `openpi_adapt_to_pi` | TongBot 通常 `false`；`true` 仅适用于 14D Aloha 关节约定 |

---

## OpenPI 后端（`pi05_backend: openpi`）

类比 ACT 的 `act_backend: aloha`：同一 `Pi05Adapter`，分支加载。

```
YAML pi05_backend: openpi
  → Pi05Adapter._load_openpi()
    → OpenpiPi05Runtime（懒 import openpi / jax）
      → create_trained_policy(get_config(name), ckpt_dir)
  → build_frame → openpi.images / openpi.state / openpi.prompt
  → policy.infer(obs) → actions [T, D]
  → expand_action_to_15d(keep_dims) → 真机
```

- **懒加载**：仅 `load()` 时 import openpi；跑 LeRobot / ACT / DP 不会拉 jax。
- **源码**：已 vendored 到 [`infer/policies/openpi`](openpi/)，无需再 clone 外部 openpi-agilex。
- **环境**：本 YAML 需在含 JAX 的 openpi 环境中运行（如 `openpi-pi05-infer`）；其它配置不受影响。
- **示例**：
  - 双臂：[`infer/config/lerobot/pi05_openpi_dual_arm_gripper_direct.yaml`](../config/lerobot/pi05_openpi_dual_arm_gripper_direct.yaml)
  - 左单臂：[`infer/config/lerobot/pi05_openpi_left_arm_direct.yaml`](../config/lerobot/pi05_openpi_left_arm_direct.yaml)
- **runtime**：[`openpi_pi05_runtime.py`](openpi_pi05_runtime.py)
- **开发者详解**（norm 回退、delta、16D 契约、J7 ±120°）：[`openpi_pi05_runtime.md`](openpi_pi05_runtime.md)

---

## 与 ACT / DP 的差异

| 项目 | Pi0.5 (LeRobot) | Pi0.5 (openpi) | ACT / DP |
|------|-----------------|----------------|----------|
| 语言输入 | ✅ `task` | ✅ `prompt` | ❌ |
| RTC | ✅ 可选 | ❌ 暂不支持 | ❌ |
| dataset_root | 不需要 | 不需要 | LeRobot 需要 stats |
| 权重格式 | LeRobot pretrained_model | openpi params / safetensors | — |

---

## 相关文件

| 文件 | 职责 |
|------|------|
| `pi05_adapter.py` | 加载、建帧、RTC / openpi 推理 |
| `openpi_pi05_runtime.py` | vendored openpi 进程内封装（懒 import） |
| `infer/core/engine.py` | 真机 sync / 发臂主循环 |
| `tools/policy/policy_batch_support.py` | 批量消融、共享 policy 重配置 |
| `common/robot_dims.py` | keep_dims / 夹爪 UDP 通道 |
