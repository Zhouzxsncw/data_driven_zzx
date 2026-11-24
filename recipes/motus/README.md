# Motus

THU-ML [Motus](https://github.com/thu-ml/Motus) 环境胶水。源码和权重不进本仓。

国内源：阿里云 PyPI、ModelScope / hf-mirror（权重）、已有 Wan2.2 软链。

| 机 | 代码 | conda | torch |
|----|------|-------|-------|
| gpu-master | `/home/xk/VLA/Motus` | `motus` | 2.9.0+cu128（FastWAM 同款，Blackwell） |
| gpu-node1 | `/data/motus/Motus` | `motus`（anaconda3） | 2.7.1+cu128（官方 pin，从 FastWAM 拷） |

## 激活

```bash
# gpu-master
source /home/xk/VLA/Motus/env.sh

# gpu-node1
source /data/motus/Motus/env.sh
```

配置：`configs/local.yaml`（Stage 1 VGM + Qwen3-VL-2B + Stage 2 SFT 起点）。

## 权重（两机已齐）

| 目录 | 大小 | 来源 |
|------|------|------|
| `Wan2.2-TI2V-5B` | 32G 软链 | 本机 DreamZero / node1 FastWAM |
| `Qwen3-VL-2B-Instruct` | 4.0G | hf-mirror |
| `Motus_Wan2_2_5B_pretrain` | 9.4G | hf-mirror Stage 1 |
| `Motus` | 15G | hf-mirror Stage 2 |

未拉 Stage 3 RoboTwin：`DOWNLOAD_ROBOTWIN=1 bash recipes/motus/link_pretrained.sh`

## 重装 / 再同步

```bash
bash recipes/motus/setup_env.sh
bash recipes/motus/link_pretrained.sh
bash recipes/motus/deploy_gpu_node1.sh
```

`flash-attn` 在 Blackwell 上未编过（走 SDPA）。训练入口：`bash recipes/motus/launch_train.sh`（单卡先把 yaml 里 `batch_size` 降到 1）。
