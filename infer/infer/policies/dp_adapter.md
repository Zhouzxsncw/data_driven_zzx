# Diffusion Policy 推理流程

对应代码：`dp_adapter.py`  
示例配置：`infer/config/lerobot/dp_left_arm_direct.yaml`  
注册类型：`policy.type: diffusion`

---

## 模型特点

- **Diffusion Policy**（LeRobot DP）：扩散模型预测动作序列
- 需要 **`dataset_root/meta/stats.json`** 归一化
- 使用相机**原始分辨率**（与数采一致），`build_frame` 仅做格式转换，不 resize
- 关键超参：`horizon`（动作序列长度）、`n_obs_steps`（观测历史帧数）
- 相机 key 可能为 `observation.images.*`（由 checkpoint 决定，与 ACT 命名可能不同）

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
1. SyncEngine.align_at_wall_time()
2. unpack 15D state + 多路 RGB
3. build_frame(images, state)
4. run_timed_inference(frame)
5. 执行 min(chunk_len, n_action_steps) 步 → expand_action_to_15d → UDP 发臂
6. 录制 machine_flow / eval_log / 视频
```

---

## Adapter 内部流程

### load()

```
DiffusionPolicy.from_pretrained(ckpt_dir)
  → resolve_image_keys_from_policy_config
  → resolve_image_target_hw
  → load_local_dataset_stats(dataset_root)
  → make_pre_post_processors(config, dataset_stats)
  → 应用 num_inference_steps_override（扩散去噪步数）
  → 记录 horizon、n_obs_steps、n_action_steps
  → policy.reset()    # 清空观测/动作缓存
```

### build_frame(images, state_15d)

与 ACT 相同模式：

| 字段 | 内容 |
|------|------|
| `observation.cam_*` 或 `observation.images.*` | `camera_image_to_model_tensor` → [1,C,H,W]（原始分辨率） |
| `observation.state` | `state_15d[keep_dims]` |

### run_timed_inference(frame)

```
preprocess(frame) → obs
  → policy.select_action(obs)    # 内部维护动作队列与观测历史
  → postprocess(raw)
  → TimedActionChunk([1或T, action_dim], 各阶段耗时)
```

---

## YAML 配置要点

| 字段 | 说明 |
|------|------|
| `type: diffusion` | 注册到 `DpAdapter` |
| `ckpt_dir` | DP checkpoint 目录 |
| `dataset_root` | 含 `meta/stats.json` 的训练数据集根目录 |
| `n_action_steps_override` | 开环步数（≤ horizon） |
| `num_inference_steps_override` | 扩散去噪迭代次数 |
| `keep_dims` | 模型动作维度映射 |

---

## 关键超参说明

| 超参 | 含义 |
|------|------|
| `horizon` | 训练时预测的动作序列长度；批量 action-step 消融上限 |
| `n_obs_steps` | 需要累积的历史观测帧数；影响可执行步数上限 |
| `n_action_steps` | 每帧实际开环执行的步数 |
| `num_inference_steps` | 扩散去噪迭代次数（越大越慢，通常更精细） |

理论开环上限：`horizon - n_obs_steps + 1`。若 `n_action_steps_override` 超过此值会打 warning。

---

## 注意事项

1. **启动校验**：DP 按 checkpoint 实际相机 key 校验（可为 `observation.images.*`），与 Pi0.5/ACT 默认 key 不同。
2. **reset_runtime**：组间切换时需 `policy.reset()` 清空观测历史，避免跨任务污染。
3. **批量推理**：支持 action-step 消融，不支持 OOD 语言阶段。

---

## 与 Pi0.5 / ACT 的差异

| 项目 | DP | ACT | Pi0.5 |
|------|-----|-----|-------|
| 模型类型 | 扩散 | Transformer chunk | VLA |
| 关键上限 | horizon | chunk_size | chunk_size |
| 观测历史 | n_obs_steps | 无 | 无 |
| 去噪步数 | num_inference_steps | — | num_inference_steps |
| 语言输入 | ❌ | ❌ | ✅ |

---

## 相关文件

| 文件 | 职责 |
|------|------|
| `dp_adapter.py` | 加载、建帧、推理 |
| `tools/policy/policy_frame_utils.py` | 图像 HWC → [1,C,H,W] 格式转换（不 resize） |
| `tools/policy/policy_observation_keys.py` | 相机 key 解析（含 DP 校验逻辑） |
| `tools/policy/lerobot_dataset_stats.py` | 离线加载 stats.json |
| `tools/camera/checkpoint_camera_resolution.py` | 按 checkpoint 设置采集分辨率 |
| `tools/policy/policy_batch_support.py` | 批量消融共用逻辑 |
