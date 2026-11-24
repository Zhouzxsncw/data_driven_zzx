# Policy 适配层

本目录是**模型注册中心**（类似 mmdetection 的 registry）：每个 `*_adapter.py` 负责一种策略的加载与推理，配合独立 YAML 即可接入真机。

```
infer/policies/
├── __init__.py         # @register_policy 注册表 + create_policy_adapter()
├── base.py             # PolicyAdapter 抽象接口
├── pi05_adapter.py     # Pi0.5（VLA，含 RTC）
├── pi05_adapter.md     # Pi0.5 推理流程说明 ←
├── act_adapter.py      # ACT
├── act_adapter.md      # ACT 推理流程说明 ←
├── dp_adapter.py       # Diffusion Policy
└── dp_adapter.md       # DP 推理流程说明 ←
```

共用工具在 `tools/policy/`；机械臂维度常量在 `common/robot_dims.py`。

---

## 各模型推理流程文档

| 模型 | Adapter | 流程文档 | 示例配置 |
|------|---------|----------|----------|
| Pi0.5 | `pi05_adapter.py` | [pi05_adapter.md](pi05_adapter.md) | LeRobot / openpi：`infer/config/lerobot/pi05_*` |
| ACT | `act_adapter.py` | [act_adapter.md](act_adapter.md) | LeRobot：`lerobot/act_*`；ALOHA ACT：`aloha/act_aloha_*`（`act_backend: aloha`） |
| Diffusion (LeRobot) | `dp_adapter.py` | [dp_adapter.md](dp_adapter.md) | `lerobot/dp_left_arm_direct.yaml` |
| ALOHA DP | `aloha_dp_adapter.py` | [aloha_dp_adapter.md](aloha_dp_adapter.md) | `aloha/dp_aloha_*`（`type: aloha_dp`） |
| ALOHA RDT | `aloha_rdt_adapter.py` | [aloha_rdt_adapter.md](aloha_rdt_adapter.md) | `aloha/rdt_aloha_*`（`type: aloha_rdt`） |

---

## 统一真机推理主链路

无论哪种模型，Native 引擎（`infer/core/engine.py`）每帧都走同一条链路：

```
infer/cli/run_infer.py
  → InferSession.run()
    → native_runner.run_native_session()
      → InferEngine.run()
```

单帧循环：

```
1. SyncEngine.align_at_wall_time()     # 多模态时间对齐（相机 + 机械臂状态）
2. unpack 15D state + 多路 RGB 图像
3. policy.build_frame(images, state[, prompt])   # 各 adapter 构造模型输入
4. policy.run_timed_inference(frame)             # preprocess → model → postprocess
5. 取 post_action chunk [T, action_dim]
6. 逐步执行 min(T, n_action_steps) 步：
     expand_action_to_15d() → UDP 发臂 → 等待反馈 → 按 fps 节拍 sleep
7. 录制 machine_flow / eval_log / 视频
```

配置入口：`infer/config/lerobot|aloha/*_direct.yaml` 中 `policy.type` 决定使用哪个 adapter。索引见 [../config/README.md](../config/README.md)。

---

## 新增模型 Checklist

1. 新建 `infer/policies/xxx_adapter.py`，继承 `PolicyAdapter`
2. 新建 `infer/policies/xxx_adapter.md`，写明推理流程
3. 类上添加 `@register_policy(PolicyType.XXX)`
4. 在 `__init__.py` 的 `_ensure_registry_loaded()` 中 import 该模块
5. 在 `infer/core/types.py` 的 `PolicyType` 枚举中增加类型（若尚未存在）
6. 编写 `infer/config/lerobot/` 或 `infer/config/aloha/` 下的 `xxx_direct.yaml`
7. 在 `startup_validation.py` 中补充 checkpoint 交叉校验（如需要）
8. 批量推理：在 `tools/policy/policy_batch_support.py` 注册支持

---

## 相关文件索引

| 模块 | 职责 |
|------|------|
| `infer/cli/run_infer.py` | CLI 入口 |
| `infer/core/session.py` | 会话编排、消融、评估后处理 |
| `infer/runners/native_runner.py` | 创建 InferEngine，多 test 循环 |
| `infer/core/engine.py` | 真机 sync / 推理 / 发臂 / 录制 |
| `infer/core/startup_validation.py` | 启动前 checkpoint 与配置校验 |
| `tools/policy/policy_frame_utils.py` | ACT/DP 图像 HWC → [1,C,H,W]（原始分辨率，不 resize） |
| `tools/policy/policy_observation_keys.py` | 相机 feature key 解析 |
| `tools/policy/lerobot_dataset_stats.py` | 离线加载 stats.json |
| `tools/policy/policy_batch_support.py` | 批量消融共用逻辑 |
| `common/robot_dims.py` | 15D 状态 / keep_dims / 夹爪 UDP 通道 |
