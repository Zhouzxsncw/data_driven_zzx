# OpenPI π₀.₅ 推理环境（TongBot）

本仓库在 `pi05_backend: openpi` 时，使用 **vendored** 的 OpenPI 源码：

[`infer/policies/openpi/`](../policies/openpi/)（来自 openpi-agilex `src/openpi`）

无需再 clone / 配置外部 `openpi-agilex` 仓库路径。YAML 只需：

- `ckpt_dir`：含 `params/` 或 `model.safetensors`（及 `assets/` norm stats）
- `openpi_config_name`：与训练一致（如 `pi05_aloha`）

配置示例：

- [`../config/lerobot/pi05_openpi_dual_arm_gripper_direct.yaml`](../config/lerobot/pi05_openpi_dual_arm_gripper_direct.yaml)
- [`../config/lerobot/pi05_openpi_left_arm_direct.yaml`](../config/lerobot/pi05_openpi_left_arm_direct.yaml)

若权重是 LeRobot `pretrained_model/`（含 `config.json`），请用 `pi05_*_direct.yaml`（默认 `pi05_backend: lerobot`），不要走本页。

---

## 环境与依赖

与 ALOHA / 普通 LeRobot 推理环境分开，推荐 conda 环境名：`openpi-pi05-infer`。

需要（与上游 openpi-agilex 一致，按本机 CUDA 调整）：

- Python ≥ 3.11
- `jax[cuda12]`、`flax`、`orbax-checkpoint`
- `torch`、`transformers`、`safetensors`
- `numpy`、`Pillow`、`einops` 等

可用上游仓库的 `uv sync` / `uv pip install -e .` 装依赖，但**运行时不再依赖该仓库源码**——本仓已自带 `infer/policies/openpi`。

---

## 启动

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
# conda activate openpi-pi05-infer

python3 infer/cli/run_batch_infer.py \
  --config infer/config/lerobot/pi05_openpi_dual_arm_gripper_direct.yaml
```

## 归一化统计（norm stats）

加载顺序（**不依赖外网也能跑**）：

1. `ckpt_dir/assets/`（若训练时写入过）
2. TrainConfig 的本地 `assets_dir`
3. 若本机已有缓存，可选用 `pi05_base`/`trossen`
4. 都没有 → **自动生成恒等统计**（quantile 下近似不做归一化），并尽量写回 `ckpt/assets/norm_stats.json`

维不匹配（如 14D Aloha stats 套 16D TongBot）会被跳过。Identity 仅当训练也是 skip-norm 时安全；训练用了真实 quantile 时必须拷入匹配的 `norm_stats.json`。

更完整的 runtime / delta / J7 说明见 [`../policies/openpi_pi05_runtime.md`](../policies/openpi_pi05_runtime.md)。

---

## 与 ALOHA vendor 的对应关系

| | ALOHA ACT/DP/RDT | OpenPI π₀.₅ |
|--|------------------|---------------|
| 源码位置 | `infer/policies/aloha_act/` | `infer/policies/openpi/` |
| 胶水 | `aloha_*_runtime.py` | `openpi_pi05_runtime.py` |
| 外仓库路径 | 无 | 无（`openpi_repo_path` 已废弃） |
| 重依赖 | torch / robomimic / diffusers | jax / flax / orbax |

---

## 与 `cigai-agilex-train` 推理的适配边界

本仓真机是 **TongBot（每臂 7 关节 + 双夹爪，16D）**，不是 AgileX Piper/Aloha（每臂 6 关节+爪，14D）。

**应对齐 agilex 的（模型调用链）：**

- `create_trained_policy` → `policy.infer(obs)`
- 相机短名 → `left_wrist_0_rgb` / `base_0_rgb` / `right_wrist_0_rgb`
- `resize_with_pad` → 224 → CHW uint8
- obs：`{state, images, prompt}`
- 训练同款 Normalize / Unnormalize / delta↔absolute（以训练 `openpi_config_name` 为准）

**必须按 TongBot 适配的（不要抄 agilex 14D，也不要抄 ACT interleaved）：**

- `openpi_action_dim: 16` + `keep_dims: [0..15]`（数采 `[左7|右7|左爪|右爪]`）
- `openpi_adapt_to_pi: false`
- `expand_action_to_15d` → RealMan / 夹爪 UDP（控制层）

库存 `pi05_aloha` 默认 delta mask 仍是 Aloha 14D `(6,-1,6,-1)`。若训练未改成 TongBot 16D mask，需在**训练配置**里改齐，不能只靠推理 YAML 硬套 14D。
