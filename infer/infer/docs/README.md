# infer 文档索引

平台仓总览：[../../../README.md](../../../README.md)。架构：[../../../docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md)。真机：[../../../docs/gpu_master_infer.md](../../../docs/gpu_master_infer.md)。

| 文档 | 读者 | 内容 |
|------|------|------|
| [../../../docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) | 开发 | Git / 网盘 / 离线机 |
| [../README.md](../README.md) | 开发 / 运维 | 推理配置、相机、产物目录、启动流程 |
| [../config/README.md](../config/README.md) | 开发 / 实习生 | 实验配置索引（lerobot / aloha） |
| [ALOHA_ENV.md](ALOHA_ENV.md) | 开发 / 运维 | 原生 ALOHA ACT/DP/RDT 环境安装（TongBot dim=8） |
| [OPENPI_ENV.md](OPENPI_ENV.md) | 开发 / 运维 | OpenPI π₀.₅（vendored）环境与配置 |
| [BATCH_INFER.md](BATCH_INFER.md) | 实习生 | 批量消融（action-step + OOD）操作手册 |
| [../policies/README.md](../policies/README.md) | 开发 | Policy 注册与统一推理链路 |
| [../policies/pi05_adapter.md](../policies/pi05_adapter.md) | 开发 | Pi0.5 推理流程 |
| [../policies/openpi_pi05_runtime.md](../policies/openpi_pi05_runtime.md) | 开发 | OpenPI π₀.₅ runtime（norm / delta / 16D / J7） |
| [REPORT_PI05_J7_SPIN_AND_INFER_ISSUES.md](REPORT_PI05_J7_SPIN_AND_INFER_ISSUES.md) | 开发 | π₀.₅ 双臂 J7 转圈与 norm 事故复盘 |
| [../policies/act_adapter.md](../policies/act_adapter.md) | 开发 | ACT（含 ALOHA ACT） |
| [../policies/aloha_dp_adapter.md](../policies/aloha_dp_adapter.md) | 开发 | 原生 ALOHA DP |
| [../policies/aloha_rdt_adapter.md](../policies/aloha_rdt_adapter.md) | 开发 | 原生 ALOHA RDT |
| [../policies/dp_adapter.md](../policies/dp_adapter.md) | 开发 | LeRobot Diffusion Policy |

## 相关文档（仓库其他目录）

| 文档 | 内容 |
|------|------|
| [../../../README.md](../../../README.md) | 平台仓总览 |
| [../../tools/README.md](../../tools/README.md) | 通用工具（policy / camera / dataset） |
| [../../../docs/FORKS.md](../../../docs/FORKS.md) | 训练 fork |
| [../../vla_eval/README.md](../../vla_eval/README.md) | VLA 自动评估 |
| [../../lower_machine/README.md](../../lower_machine/README.md) | 下位机 |
| [../../../collect/README.md](../../../collect/README.md) | 数据采集 |

## 常用命令（文档内路径已对齐）

```bash
cd /path/to/data_Driven
export PYTHONPATH="$(pwd)/legacy/CQTongBot_infer/src:$(pwd)/infer:${PYTHONPATH:-}"
cd infer

python3 infer/cli/run_infer.py --config infer/config/lerobot/pi05_left_arm_direct.yaml
python3 infer/cli/run_batch_infer.py
python3 infer/cli/run_eval.py --test-dir /path/to/action-step_10
python3 -m pytest infer/tests/test_startup_validation.py -v
```

真机日常请用 [`recipes/`](../../../recipes/README.md)，不要把上面旧单臂 YAML 当默认。
