# recipes

各模型 **启动胶水**：起 websocket、指 ckpt、写 TongBot YAML 快照。  
**不包含**模型实现（那在对应 fork 的 `tongbot` 分支）和权重。

真机逐步操作见 [`docs/gpu_master_infer.md`](../docs/gpu_master_infer.md)。

| 目录 | 默认任务 | 端口 |
|------|----------|------|
| `pi05/` | 00086 | 无（单进程） |
| `lingbot_vla/` | 00086 | 8006 |
| `gr00t/` | 00073 TongBot 16D | 5555 |
| `gr00t_sonic/` | G1 SONIC 采集/训练/仿真+真机（`UNITREE_G1_SONIC`） | **5550** |
| `lingbot_va/` | 00036 / 00073 | 1234 |
| `dreamzero/` | 00073 | 1235 |
| `cosmos3/` | 00073 SFT（[推理参数](../docs/COSMOS_TAU_INFER_PARAMS.md)） | 1236 |
| `internvla/` | 00036 | 1237 |
| `memoryvla/` | 00086 | 1238 |
| `walloss/` | 00086 EE（[数据流](../docs/WALLOSS_DATAFLOW.md)，[一整套说明](walloss/README.md)，[IK mode/solver](walloss/IK.md)） | 1239 |
| `fastwam/` | 00036 本机 / 00073 gpu-node1 | 训练（无真机端口） |
| `motus/` | 环境 / 预训练 / gpu-node1 | 训练（无真机端口） |
| `gigaworld_policy/` | GWP-0.5 00073 TongBot | **1243**（官方 ZMQ open-loop 仍是 11444） |
| `tau0_wm/` | τ0-WM 00036 EE 20D（[推理参数](../docs/COSMOS_TAU_INFER_PARAMS.md)） | **1244**（官方 openpi 协议不要给真机客户端） |

真机客户端统一入口：

```bash
bash recipes/common/launch_client.sh gwp
bash recipes/common/launch_client.sh --list
```

脚本里服务端 `INFER_ROOT` 默认 `/home/xk/VLA/infer`。只用本仓：

```bash
export INFER_ROOT=/path/to/data_Driven/infer
bash recipes/<model>/launch_*_server.sh
```
