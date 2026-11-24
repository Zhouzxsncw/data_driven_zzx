# 原生 ALOHA Diffusion Policy

对应代码：`aloha_dp_adapter.py` / `aloha_dp_runtime.py`  
示例：`infer/config/aloha/dp_aloha_left_arm_direct.yaml`  
注册：`policy.type: aloha_dp`

与 LeRobot `dp_adapter.py`（`type: diffusion`）完全独立。

## 需要什么

| 项 | 说明 |
|----|------|
| 权重 | `policy_best.ckpt` = `{nets, ema}`（Diffusion.serialize） |
| 统计 | 同目录 `dataset_stats.pkl` |
| TongBot 维数 | `arm_joint_state_dim: 8`（已写在 YAML） |
| 依赖 | 见 [../docs/ALOHA_ENV.md](../docs/ALOHA_ENV.md)：`robomimic` + `diffusers` + ACT 公共包 |
| 关键 args | `chunk_size`, `num_inference_timesteps`, `ema_power`, `use_transformer`, 相机与 arm 维 |

观测预处理与 ALOHA ACT 相同（共享 `aloha_common`），模型构建走 vendored `DiffusionPolicy`。
