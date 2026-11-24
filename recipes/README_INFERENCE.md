# 全模型推理服务端与客户端启动指南

本文档涵盖所有已集成模型的**服务端**和**客户端**启动命令。

---

## 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│  服务端 (Server)                                             │
│  加载模型权重，监听 WebSocket 端口，执行 forward 推理         │
│  每个模型独立端口，可同时启动多个服务端                       │
└─────────────────────────────────────────────────────────────┘
                            ↑ WebSocket (msgpack_numpy)
                            │
┌─────────────────────────────────────────────────────────────┐
│  客户端 (Client)                                             │
│  采集相机 + 读取机器人状态 → 发送 obs → 接收 action → 下发   │
│  ⚠️ 同一时刻只能有一个客户端连接真机硬件                      │
└─────────────────────────────────────────────────────────────┘
```

**启动流程**：
1. 先启动服务端：`bash recipes/<model>/launch_*_server.sh`
2. 服务进程起来后会自动 SSH 到 `lab@192.168.1.13` 拉起夹爪 + 外骨骼（`run.py`）。下位机不通会告警，但不阻断模型服务。跳过：`SKIP_LOWER_MACHINE=1`。单独重试：`bash recipes/common/start_lower_machine.sh`
3. 再启动客户端（**只换 YAML**，真机进程是同一套）：
   ```bash
   bash recipes/common/launch_client.sh gwp
   bash recipes/common/launch_client.sh --list
   ```
   旧的 `recipes/<model>/launch_*_client.sh` 仍可用，内部转到上面这条。

---

## 端口总表

| 模型 | 端口 | Conda 环境 | 状态 |
|------|------|-----------|------|
| **ACoT-VLA** | `8000` | ACoT-VLA venv | ✅ 可用 |
| **LingBot-VA** | `1234` | lingbotva | ✅ 可用 |
| **DreamZero** | `1235` | dreamzero | ✅ 可用 |
| **Cosmos3** | `1236` | cosmos3 | ✅ 可用 |
| **InternVLA** | `1237` | internvla | ✅ 可用 |
| **MemoryVLA** | `1238` | memoryvla | ✅ 可用 |
| **WALL-OSS** | `1239` | walloss | ✅ 可用 |
| **FastWAM** | `1240` | fastwam | ✅ 可用 |
| **Motus** | `1241` | motus | ✅ 可用 |
| **GR00T** | `5555` | gr00t_n17 | ✅ 可用 |
| **LingBot-VLA** | `8006` | lingbotvla | ✅ 可用 |
| **π₀.₅ (OpenPI)** | `1242` | 服务端 jax venv / 客户端 lingbotva | ✅ 可用 |
| **GigaWorld-Policy-0.5** | `1243` | gigaworld-policy | ⚠️ 代码已接，需拉 00073 `transformer/` |
| **τ0-WM** | `1244` | tau0wm | ⚠️ 代码已接，等 00036 EE `step_N`（约 11G） |

---

## 各模型启动命令

### 1. ACoT-VLA

**模型**：Action Chain-of-Thought VLA（JAX，PaliGemma 2B + 双 gemma expert）

**数据集**：00036（香蕉柠檬放黑盒，蔬菜放蓝盒）

```bash
# 服务端（端口 8000）
bash recipes/acot_vla/launch_acot_server.sh

# 客户端
bash recipes/acot_vla/launch_acot_client.sh
```

**日志**：
```bash
tail -f /home/xk/ACoT-VLA/serve_acot.log
```

**Checkpoint**：`/home/xk/ACoT-VLA/checkpoints/acot_tongbot_00036/tongbot_00036_run1/4000`

---

### 2. FastWAM

**模型**：FastWAM 5B video expert + 1.02B action expert

**数据集**：00073（快递分拣）

```bash
# 服务端（端口 1240）
bash recipes/fastwam/launch_fastwam_server.sh

# 客户端
bash recipes/fastwam/launch_fastwam_client.sh
```

**日志**：
```bash
tail -f recipes/fastwam/server_fastwam_00073.log
```

**Checkpoint**：`/home/xk/VLA/FastWAM/runs/tongbot_00073_3cam384_1e-4/2026-09-07_19-14-02/checkpoints/weights/step_020000.pt`

---

### 3. LingBot-VA

**模型**：LingBot Video-Action（扩散策略）

**数据集**：00036 / 00073

```bash
# 服务端 00036（端口 1234）
bash recipes/lingbot_va/launch_00036_server.sh

# 客户端 00036
bash recipes/lingbot_va/launch_00036_client.sh

# 或 00073
bash recipes/lingbot_va/launch_00073_server.sh
bash recipes/lingbot_va/launch_00073_client.sh
```

**Checkpoint**：
- 00036：`/home/xk/VLA/h100_ckpts/00036_va_w30/checkpoint_step_10000/`
- 00073：`/home/xk/VLA/h100_ckpts/00073_va_16gpu_w30/checkpoint_step_500/`

---

### 4. LingBot-VLA

**模型**：LingBot VLA（Qwen3-VL 2B + action expert）

**数据集**：00086（抽屉药盒任务）

```bash
# 服务端（端口 8006）
bash recipes/lingbot_vla/start_lingbot_vla_server.sh

# 客户端（batch infer 模式）
bash recipes/lingbot_vla/start_lingbot_vla_batch_infer.sh
```

**Checkpoint**：`/home/xk/VLA/h100_ckpts/00086_lingbotvla/checkpoint/9287/hf_ckpt`

---

### 5. Cosmos3

**模型**：Cosmos3 视频生成 + 动作预测

**数据集**：00036 / 00073 / 00091

```bash
# 服务端 00036（端口 1236）
bash recipes/cosmos3/launch_cosmos3_00036_server.sh

# 客户端 00036
bash recipes/cosmos3/launch_cosmos3_00036_client.sh

# 或 00073
bash recipes/cosmos3/launch_cosmos3_server.sh
bash recipes/cosmos3/launch_cosmos3_client.sh

# 或 00091
bash recipes/cosmos3/launch_cosmos3_00091_server.sh
bash recipes/cosmos3/launch_cosmos3_00091_client.sh
```

**Checkpoint**：
- 00036：`/home/xk/VLA/h100_ckpts/cosmos3_00036/iter_000001500`
- 00073：`/home/xk/VLA/h100_ckpts/cosmos3_00073/iter_000001500`
- 00091：`/home/xk/VLA/h100_ckpts/cosmos3_00091/iter_000001500`

---

### 6. DreamZero

**模型**：DreamZero 5B（强化学习 + 视频想象）

**数据集**：00073

```bash
# 服务端（端口 1235）
bash recipes/dreamzero/launch_dreamzero_server.sh

# 客户端
bash recipes/dreamzero/launch_dreamzero_client.sh
```

**Checkpoint**：`/home/xk/VLA/h100_ckpts/dreamzero_5b/global_step_3000/actor/model_state_dict/infer_weights.pt`

---

### 7. InternVLA

**模型**：InternVLA（Qwen3.5-2B + action head）

**数据集**：00036 / 00086

```bash
# 服务端（端口 1237）
bash recipes/internvla/launch_internvla_server.sh

# 客户端
bash recipes/internvla/launch_internvla_client.sh
```

**Checkpoint**：`/data/internvla/outputs/00036_finetune_bs32/checkpoints/002000/pretrained_model`

---

### 8. MemoryVLA

**模型**：MemoryVLA（长时记忆 VLA）

**数据集**：00086

```bash
# 服务端（端口 1238）
bash recipes/memoryvla/launch_memoryvla_server.sh

# 客户端
bash recipes/memoryvla/launch_memoryvla_client.sh
```

**Checkpoint**：`/data/memoryvla/outputs/00086/memvla_00086_80k--image_aug/checkpoints/step-010000-epoch-00-loss=0.0135.pt`

---

### 9. WALL-OSS

**模型**：WALL-OSS 0.5（开源 VLA）

**数据集**：00086 / 00091

```bash
# 服务端（端口 1239）
bash recipes/walloss/launch_walloss_server.sh

# 客户端
bash recipes/walloss/launch_walloss_client.sh
```

**Checkpoint**：`/data/wall-oss/outputs/walloss05_tongbot_ee_00086/9`

---

### 10. Motus

**模型**：Motus（THU-ML，视频想象 + 动作）

**数据集**：00073

```bash
# 服务端（端口 1241）
bash recipes/motus/launch_motus_server.sh

# 客户端
bash recipes/motus/launch_motus_client.sh
```

**Checkpoint**：`/home/xk/VLA/h100_ckpts/motus_tongbot_jiangsuan4/checkpoint_step_7000`

---

### 10b. GigaWorld-Policy-0.5

**模型**：GWP-0.5（Wan2.2-TI2V-5B MoT，TongBot 00073 微调）

**数据集**：00073（快递分拣）

```bash
# 先拉推理权重（只要 transformer/ 约 12G，不要 79G optimizer）
bash recipes/gigaworld_policy/pull_gwp_ckpt.sh

# 服务端（端口 1243）
bash recipes/gigaworld_policy/launch_gwp_server.sh

# 客户端（等日志出现 listening）
bash recipes/gigaworld_policy/launch_gwp_client.sh

# 只加载权重
bash recipes/gigaworld_policy/launch_gwp_server.sh --load-only
```

**日志**：
```bash
tail -f recipes/gigaworld_policy/server_gwp.log
```

**Checkpoint**：`/home/xk/VLA/h100_ckpts/gwp05_tongbot_00073/transformer/`（H100 终档 `checkpoint_epoch_22_step_50000`）

菜单里 action-step **必须是 48**。三路相机 T 形 320×384，14 关节 delta + 2 夹爪绝对。

---

### 10c. τ0-WM

**模型**：τ0-WM VAM-5B（Wan2.2-TI2V-5B + Action DiT），TongBot 00036 **EE 20D** 后训练

**数据集**：00036（香蕉柠檬放黑盒，蔬菜放蓝盒）

```bash
# 训练出 step_N 后再拉推理权重（只要 diffusion_pytorch_model.bin ≈ 11G，不要 optimizer）
bash recipes/tau0_wm/pull_tau_ckpt.sh

# 服务端（端口 1244）
bash recipes/tau0_wm/launch_tau_server.sh

# 客户端（等日志出现 listening）
bash recipes/tau0_wm/launch_tau_client.sh

# 只加载权重 / 先用发布预训练冒烟
bash recipes/tau0_wm/launch_tau_server.sh --load-only
bash recipes/tau0_wm/launch_tau_server.sh --pretrained-only
```

**日志**：
```bash
tail -f recipes/tau0_wm/server_tau0wm.log
```

**Checkpoint**：`/home/xk/VLA/h100_ckpts/tau0wm_tongbot_00036_ee/`（H100 `.../tau0wm_vam5b_ee_rela_192x256_bs1/step_N`）

菜单里 action-step 范围 **1~33**，所选值 = 服务端 IK 路点数。三路各自 **192×256**。rot6d **列约定**（不要用 WALL-OSS 行约定）。客户端收 16D 关节。

---

### 11. GR00T

**模型**：NVIDIA GR00T N1.7

**数据集**：00036 / 00073

```bash
# 服务端（端口 5555）
bash recipes/gr00t/start_gr00t_server.sh

# 客户端（batch infer）
bash recipes/gr00t/start_gr00t_batch_infer.sh
```

---

### 12. π₀.₅ (OpenPI)

**模型**：π₀.₅（Physical Intelligence，JAX）

**数据集**：00092（药盒/抽屉；也可换 00086）

**服务端**（JAX venv，加载权重，端口 1242）  
**客户端**（lingbotva，相机 + 机械臂，不需要 JAX）

```bash
# 真机 TongBot（默认，端口 1242）——同一份 YAML
bash recipes/pi05/launch_pi05_server.sh
bash recipes/pi05/launch_pi05_client.sh

# LIBERO sim：只改 embodiment，维数/相机会从 YAML 自动套上
PI05_EMBODIMENT=libero bash recipes/pi05/launch_pi05_server.sh
PI05_EMBODIMENT=libero bash recipes/pi05/launch_pi05_client.sh
# 或改 YAML 里 policy.openpi_embodiment: libero 后直接 launch_* 即可
bash recipes/pi05/launch_pi05_libero_client.sh --fake-obs   # 不启仿真，只验握手/维数

# gpu-node1 / 江算 / 华为：同一份 infer/recipes，路径只在远端 host_env.sh
#   bash scripts/sync_infer_code.sh
#     默认推 jiangsuan-1、4、16 和华为 A3(16×910C) / A2(8×910B)
# js16 容器: docker exec -it openpi_train bash -lc 'cd /workspace/openpi/data_Driven && PI05_EMBODIMENT=libero bash recipes/pi05/launch_pi05_server.sh'
# 华为容器: docker exec -it pi05 bash -lc 'cd /home/pi05/data_Driven && PI05_EMBODIMENT=libero bash recipes/pi05/launch_pi05_server.sh'

# 只加载权重（不监听端口）
bash recipes/pi05/launch_pi05_server.sh --load-only
```

**日志**：
```bash
tail -f recipes/pi05/server_pi05_00092.log
```

**Checkpoint**：`/home/xk/VLA/h100_ckpts/00092_pi05_jiangsuan1/012570/pretrained_model`

单进程回退（模型 + 相机 + 机械臂都在 JAX venv）：
```bash
bash recipes/pi05/start_pi05_infer.sh
```

---

## 环境变量

可通过环境变量覆盖默认配置：

```bash
# ACoT-VLA
export ACOT_PORT=8000
export ACOT_PYTHON=/path/to/python

# FastWAM
export FASTWAM_PORT=1240
export FASTWAM_PYTHON=/path/to/python
export FASTWAM_ROOT=/home/xk/VLA/FastWAM

# Cosmos3
export COSMOS3_PORT=1236
export COSMOS3_CKPT=/path/to/checkpoint
export COSMOS3_STEPS=5

# DreamZero
export DREAMZERO_PORT=1235
export DREAMZERO_CKPT=/path/to/checkpoint

# InternVLA
export INTERNVLA_PORT=1237
export INTERNVLA_CKPT=/path/to/checkpoint

# MemoryVLA
export MEMORYVLA_PORT=1238
export MEMVLA_CKPT=/path/to/checkpoint

# WALL-OSS
export WALLOSS_PORT=1239
export WALLOSS_CKPT=/path/to/checkpoint

# Motus
export MOTUS_PORT=1241
export MOTUS_PYTHON=/path/to/python
export MOTUS_CKPT=/path/to/checkpoint

# GigaWorld-Policy-0.5
export GWP_PORT=1243
export GWP_CKPT=/home/xk/VLA/h100_ckpts/gwp05_tongbot_00073
export GWP_NORM_STATS=/home/xk/VLA/h100_ckpts/gwp05_tongbot_00073/norm_stats_00073.json

# τ0-WM
export TAU_PORT=1244
export TAU_CKPT=/home/xk/VLA/h100_ckpts/tau0wm_tongbot_00036_ee
export TAU_N_EXEC=8

# LingBot-VA
export LINGBOTVA_PYTHON=/path/to/python

# LingBot-VLA
export LINGBOTVLA_PYTHON=/path/to/python

# GR00T
export GR00T_DIR=/path/to/gr00t

# π₀.₅
export PI05_CKPT=/path/to/checkpoint
export PI05_PYTHON=/path/to/jax/python          # 服务端
export PI05_CLIENT_PYTHON=/path/to/lingbotva/python
export PI05_PORT=1242
```

---

## 推理电脑（cigai20）

| 项目 | 值 |
|------|----|
| hostname | `cigai20-thinkstation-p3-tower-gen-2` |
| 局域网 IP（有线，优先） | `192.168.1.12` |
| 广域网 IP（备用） | `10.151.46.4` |
| 用户名 | `cigai22` |
| HOME | `/home/cigai22` |
| 推理代码路径 | `/home/cigai22/infer/` |
| GPU | RTX 4090 24GB |

```bash
# 连接（局域网，快）
ssh cigai22@192.168.1.12

# 连接（广域网，慢）
ssh -F /home/xk/VLA/ssh_config gpu-cigai-20
```

---

## 硬件连接

客户端连接的硬件端口（所有模型共用）：

| 设备 | 端口 | 说明 |
|------|------|------|
| 上位机 (high_host) | `192.168.1.14` | gpu-master 本机 |
| 下位机 (low_host) | `192.168.1.13` | 夹爪/传感器 |
| 右臂 | `192.168.1.17:8080` | RM 机械臂 |
| 左臂 | `192.168.1.18:8080` | RM 机械臂 |
| 夹爪控制 | `30015` | Dynamixel UDP |
| 夹爪状态 | `30016` | Dynamixel UDP |
| 相机 | `18888` | RealSense L515 ×3 |

**相机序列号**：
- front: `f1420947`
- left: `f1480854`
- right: `f1380315`

---

## 常见问题

### Q: 多个服务端可以同时运行吗？
**可以**。每个模型用不同端口，服务端互不干扰。但 GPU 显存有限，同时跑多个会 OOM。

### Q: 可以同时开多个客户端吗？
**不可以**。同一时刻只能有一个客户端连接真机硬件（相机、机械臂、夹爪）。

### Q: 服务端启动失败？
检查：
1. 端口是否被占用：`ss -tln | grep <PORT>`
2. 日志：`tail -f recipes/<model>/*.log`
3. Conda 环境是否正确：`which python`

### Q: 客户端连不上服务端？
1. 确认服务端已启动：`ss -tln | grep <PORT>`
2. 确认端口一致（yaml 里的 `lingbot_port` 与服务端端口匹配）

### Q: π₀.₅ 怎么拆服务端和客户端？
先 `launch_pi05_server.sh`（JAX，端口 1242），再 `launch_pi05_client.sh`（lingbotva）。YAML 里 `pi05_backend: websocket`。真机和 LIBERO 共用 `pi05_openpi_tongbot_16d_00092_ws_client.yaml`：改 `policy.openpi_embodiment` 或设 `PI05_EMBODIMENT=libero` 即可切换，维数/相机/keep_dims 由加载器填。LIBERO 会强制关掉真机 IO；`launch_pi05_client.sh` 在 libero 时自动转到 `eval_libero_client.py`，避免 7D 动作下发 TongBot。单进程回退仍可用 `start_pi05_infer.sh`。

---

## 文件结构

```
recipes/
├── common/
│   ├── launch_client.sh          # 统一真机客户端（只换 YAML）
│   └── start_lower_machine.sh
├── acot_vla/
│   ├── launch_acot_server.sh
│   └── launch_acot_client.sh
├── fastwam/
│   ├── launch_fastwam_server.sh
│   ├── launch_fastwam_client.sh
│   └── serve_tong_fastwam.py
├── lingbot_va/
│   ├── launch_00036_server.sh
│   ├── launch_00036_client.sh
│   ├── launch_00073_server.sh
│   └── launch_00073_client.sh
├── lingbot_vla/
│   ├── start_lingbot_vla_server.sh
│   └── start_lingbot_vla_batch_infer.sh
├── cosmos3/
│   ├── launch_cosmos3_*_server.sh
│   └── launch_cosmos3_*_client.sh
├── dreamzero/
│   ├── launch_dreamzero_server.sh
│   └── launch_dreamzero_client.sh
├── internvla/
│   ├── launch_internvla_server.sh
│   └── launch_internvla_client.sh
├── memoryvla/
│   ├── launch_memoryvla_server.sh
│   └── launch_memoryvla_client.sh
├── walloss/
│   ├── launch_walloss_server.sh
│   └── launch_walloss_client.sh
├── motus/
│   ├── launch_motus_server.sh
│   └── launch_motus_client.sh
├── gr00t/
│   ├── start_gr00t_server.sh
│   └── start_gr00t_batch_infer.sh
└── pi05/
    ├── launch_pi05_server.sh
    ├── launch_pi05_client.sh          # 按 YAML embodiment 分发：真机或 LIBERO
    ├── launch_pi05_libero_client.sh   # LIBERO sim（也可被 launch_pi05_client.sh 调用）
    ├── eval_libero_client.py
    ├── serve_tong_pi05.py
    └── start_pi05_infer.sh   # 单进程回退

infer/infer/
├── config/lingbot/          # 客户端 YAML 配置
│   ├── acot_vla_tongbot_16d_00036_local.yaml
│   ├── fastwam_tongbot_16d_00073_local.yaml
│   └── ...
└── policies/                # Policy adapter
    ├── acot_vla_adapter.py
    ├── fastwam_adapter.py
    └── ...
```

---

**最后更新**：2026-09-10（π₀.₅ 拆成 1242 服务端 + 客户端）
