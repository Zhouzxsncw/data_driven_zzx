# Cosmos3 TongBot 推理

端口 **1236**，握手 `cosmos3_nano`。真机启动逐步表：[docs/gpu_master_infer.md](../../docs/gpu_master_infer.md) §10。

N 卡 / 华为 / 江算 **环境怎么配**（独立 Docker、SDPA、Nano 基座、踩坑）：[docs/AI workflow.md](../../docs/AI%20workflow.md)「预训练环境」· 知识库 [docs/kb/pretrained_env.yaml](../../docs/kb/pretrained_env.yaml)。

**推理超参（denoise / guidance / flow_shift / 钉死的 action-step）** 与 τ0-WM 对照：

[docs/COSMOS_TAU_INFER_PARAMS.md](../../docs/COSMOS_TAU_INFER_PARAMS.md)

```bash
bash recipes/cosmos3/setup_env.sh          # 只需一次
bash recipes/cosmos3/launch_cosmos3_server.sh --load-only
bash recipes/cosmos3/launch_cosmos3_server.sh
bash recipes/cosmos3/launch_cosmos3_client.sh
```

00036 / 00091 用 `launch_cosmos3_00036_*.sh`、`launch_cosmos3_00091_*.sh`。扫 denoise×action-step 用 `launch_cosmos3_grid.sh`。
