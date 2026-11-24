# infer 实验配置索引

按训练 / 推理框架分子目录，避免扁平目录里找不到文件。

```text
infer/config/
├── local_env.yaml                     # 本机 IP / 相机 / SDK（共享）
├── ood_language_presets.yaml
├── ood_language_presets_dual_arm.yaml
├── lerobot/                           # LeRobot / OpenPI
└── aloha/                             # 原生 ALOHA ACT / DP / RDT（TongBot）
```

## 怎么选

| 你训练用的是… | 去哪个目录 | `policy.type` |
|---------------|------------|---------------|
| LeRobot / OpenPI | [`lerobot/`](lerobot/) | `pi05` / `act` / `diffusion` |
| 原生 ALOHA ACT（TongBot） | [`aloha/`](aloha/) | `act` + `act_backend: aloha` |
| 原生 ALOHA DP（TongBot） | [`aloha/`](aloha/) | `aloha_dp` |
| 原生 ALOHA RDT（TongBot） | [`aloha/`](aloha/) | `aloha_rdt` |

## ALOHA / TongBot

**维数约定（本目录配置已写死）：** `arm_joint_state_dim: 8`（7 关节 + 夹爪）。左臂 8 维 / 双臂 16 维。训练 TongBot ckpt 时请用同一维数。

| 模型 | Adapter | 配置 | 权重 | 环境依赖 |
|------|---------|------|------|----------|
| ACT | `ActAdapter` + `AlohaActRuntime` | `act_aloha_*` | flat `policy_best.ckpt` | torch + opencv 等（**不需要** timm） |
| DP | `AlohaDpAdapter` + `AlohaDpRuntime` | `dp_aloha_*` | `{nets, ema}` | ACT 依赖 + **robomimic** + **diffusers** |
| RDT | `AlohaRdtAdapter` + `AlohaRdtRuntime` | `rdt_aloha_*` | flat state_dict | ACT 依赖 + **diffusers** + **timm** |

**环境怎么装、怎么自检：** 见 [../docs/ALOHA_ENV.md](../docs/ALOHA_ENV.md)（推荐独立 conda 环境 `aloha-infer`）。

```bash
conda activate aloha-infer
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 infer/cli/run_batch_infer.py --config infer/config/aloha/act_aloha_left_arm_direct.yaml
python3 infer/cli/run_batch_infer.py --config infer/config/aloha/dp_aloha_left_arm_direct.yaml
python3 infer/cli/run_batch_infer.py --config infer/config/aloha/rdt_aloha_dual_arm_gripper_direct.yaml
```

## LeRobot（摘要）

见 [../README.md](../README.md)。默认推荐：[`lerobot/pi05_left_arm_direct.yaml`](lerobot/pi05_left_arm_direct.yaml)。

OpenPI 原生 ckpt（`params/` / `model.safetensors`）用 `pi05_openpi_*.yaml`（`pi05_backend: openpi`）；源码已在 [`../policies/openpi/`](../policies/openpi/)，环境见 [../docs/OPENPI_ENV.md](../docs/OPENPI_ENV.md)。
