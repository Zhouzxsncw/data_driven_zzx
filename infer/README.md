# infer

TongBot **真机客户端**（相机、双臂、adapter、`pipeline_record`）。模型 forward 不在这里，在各 fork 的 `tongbot` 分支；本仓只负责起服务的胶水在 [`recipes/`](../recipes/README.md)。

## 从哪进

| 你要做什么 | 用这个 |
|------------|--------|
| 真机推理（日常） | 仓库根 [`docs/gpu_master_infer.md`](../docs/gpu_master_infer.md) + `recipes/<model>/` |
| 客户端模块 / YAML | [`infer/README.md`](infer/README.md)（本目录下的 `infer/` 包） |
| 外骨骼数采 | 仓库根 [`collect/README.md`](../collect/README.md)（实现软链到这里） |
| 下位机脚本 | [`lower_machine/README.md`](lower_machine/README.md) |

```bash
cd /path/to/data_Driven
export PYTHONPATH="$(pwd)/legacy/CQTongBot_infer/src:$(pwd)/infer:${PYTHONPATH:-}"
```

不要直接 `python infer/cli/run_batch_infer.py` 开第二条客户端：硬件只有一套。π₀.₅ 是单进程，其它模型先起 `recipes/*/launch_*_server.sh`。

权重、`h100_ckpts/`、上游训练源码都不在本目录。
