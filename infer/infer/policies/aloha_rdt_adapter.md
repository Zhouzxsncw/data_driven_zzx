# 原生 ALOHA RDT

对应代码：`aloha_rdt_adapter.py` / `aloha_rdt_runtime.py`  
示例：`infer/config/aloha/rdt_aloha_left_arm_direct.yaml`  
注册：`policy.type: aloha_rdt`

## 需要什么

| 项 | 说明 |
|----|------|
| 权重 | flat `policy_best.ckpt`（`RTPolicy.state_dict`） |
| 统计 | 同目录 `dataset_stats.pkl` |
| TongBot 维数 | `arm_joint_state_dim: 8`（已写在 YAML） |
| 依赖 | 见 [../docs/ALOHA_ENV.md](../docs/ALOHA_ENV.md)：`diffusers` + `timm` + ACT 公共包 |
| 关键 args | `use_rdt: true`, `use_diffusion: true`, `num_inference_timesteps`（须与训练一致，运行时会读 YAML）, `transformer_layer_num`, `chunk_size`, 相机与 arm 维 |

`use_future` 暂不支持。观测预处理与 ALOHA ACT 相同。
