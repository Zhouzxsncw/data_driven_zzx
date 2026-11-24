# PI0-FAST TongBot 训练与推理配方

OpenPI π₀.5 **FAST**（离散动作 token 自回归，无 action expert / flow-matching）在 TongBot
双臂数据上的全参微调 + 推理服务。2026-09-10 部署。

## 布局

| 机器 | 任务 | 数据集 | 训练 run |
|---|---|---|---|
| gpu-master（本机, RTX PRO 6000 96GB） | 00036 全参训练 + 推理服务 | `/home/xk/dataset/00036_20260804_RM_01_yj_v21` | `checkpoints/00036_pi0fast_bs16_lr2.5e-5_ah15_1gpu` |
| gpu-node1（10.151.43.124, 96GB） | 00073 全参训练（watcher 等 X-VLA 退出后自动启动） | `/home/cqcigai/datasets/00073_20260818_RM_01_yj_21` | `checkpoints/00073_pi0fast_bs32_lr2.5e-5_ah15_1gpu` |

训练仓库：`/home/xk/VLA/openpi/cigai-agilex-train-openpi`（gpu-master）、
`/home/cqcigai/cigai-agilex-train-openpi`（gpu-node1）。

## 关键决策（与 PI05 配方的差异）

1. **action_horizon=15 / max_token_len=280**：FAST 把整个动作 chunk 压成 DCT+BPE token，
   实测 16D×15 ≈ 250 token（含 prompt+state 前缀）；horizon 50 需 572 token 超出 PaliGemma
   预算。官方双臂建议 ~250，取 280 留余量。
2. **action_dim=16 严格匹配**：PI0-FAST 数据管线（`ModelTransformFactory` PI0_FAST 分支）
   无 `PadStatesAndActions`，不能用 32D pad——预设 `pi0_fast_tongbot` 定死 16。
3. **delta_action_dims=(7,7,-1,-1)**：左臂 7 关节 delta | 右臂 7 关节 delta | 双爪绝对值
   （与 PI05 TongBot 配方一致；数据 action 布局 ljoint1-7|joint1-7|gripper|right_gripper）。
4. **JAX framework**：`models_pytorch/` 无 FAST 实现，只有 JAX `pi0_fast.py`。
5. **权重加载**：预设走 `gs://openpi-assets/checkpoints/pi0_fast_base/params`，
   `maybe_download` 本地缓存优先 → 预置 `~/.cache/openpi/openpi-assets/checkpoints/pi0_fast_base/`
   （**注意：不能用 symlink**，`resolve()` 后 `relative_to(cache_dir)` 会崩；用 `cp -al` 硬链接）。
6. **FAST tokenizer**：HF `physical-intelligence/fast`（DCT+BPE，~1MB），缓存在
   `~/.cache/huggingface/hub/models--physical-intelligence--fast/`，服务/训练均
   `HF_HUB_OFFLINE=1`。
7. **本机显存编排**：训练 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`（~73GB，全参
   params+EMA+AdamW m/v 静态 ~41GB + bs16 activations），推理服务 0.2（~19GB）共卡。
   （motus 推理服务已停，重启命令见本文件末尾。）
8. **视频解码**：本机 venv torchcodec 0.4.0 与 torch 2.8 ABI 不匹配（ops 未注册），
   已卸载 → lerobot 自动回退 **pyav**（av 14.4.0）。node1 是 torch 2.11 + torchcodec
   0.11.1 正常。

## 预设注册（三处 config.py，幂等补丁）

```bash
python3 recipes/pi0fast/train/patch_config_fast_tongbot.py \
    --repo /home/xk/VLA/openpi/cigai-agilex-train-openpi          # 本机训练仓库
python3 recipes/pi0fast/train/patch_config_fast_tongbot.py \
    --config-path /home/xk/data_Driven/infer/infer/policies/openpi/training/config.py  # vendored 推理
python3 recipes/pi0fast/train/patch_config_fast_tongbot.py \
    --repo /home/xk/openpi/cigai-agilex-train-openpi              # venv editable 命中的副本（推理实际 import 路径）
```

注册 `pi0_fast_tongbot`（全参）与 `pi0_fast_tongbot_lora`（低显存备选，EMA 关闭）。
原文件备份为 `config.py.orig_fast_tongbot`。

> 注意：本机 openpi venv 里有 editable install 指向 `/home/xk/openpi/cigai-agilex-train-openpi`
> （PEP-660 import hook 优先级高于 sys.path），推理必须给这个副本也打补丁。

## 训练

```bash
# 本机 00036（bs16, 20000 step, ~12h）
bash recipes/pi0fast/train/start_train_00036_gpu_master.sh
tail -f /home/xk/VLA/openpi/train_pi0fast_00036.log

# 本机从 node1 ckpt-10000 热启动 00073（bs16；10000 无 train_state，不能 Orbax --resume）
bash recipes/pi0fast/train/start_train_00073_gpu_master.sh
tail -f /home/xk/VLA/openpi/train_pi0fast_00073_from10000.log
# ckpt: cigai-agilex-train-openpi/checkpoints/00073_pi0fast_bs16_from10000
# 保存：每 10000 步 params-only，终盘才写 full train_state
```

超参：peak_lr 2.5e-5（官方 FAST 微调默认）、warmup 1000、cosine decay 至 2.5e-6、
bs16/32、EMA 0.99、save_interval 2000（params-only 轻量）、save_full_interval 8000/10000
（含 optimizer）、keep_period 4000。

## 推理（本机 gpu-master）

```bash
# 1) 服务端（端口 1243，与训练共卡；自动解析 run 最新 step 的 pretrained_model）
bash recipes/pi0fast/launch_pi0fast_server.sh
tail -f recipes/pi0fast/server_pi0fast_00036.log

# 2) 客户端（lingbotva env，无 JAX；相机+双臂）
bash recipes/pi0fast/launch_pi0fast_client.sh
```

- 服务端协议与 PI05 一致（msgpack_numpy over websocket）：handshake
  `{model: pi0_fast, action_horizon: 15, action_dim: 16, rtc_capable: false}`；
  infer `{obs:{state,images,prompt}}` → `{actions:(15,16)}`。**FAST 无 RTC**
  （离散自回归解码，无 flow 前缀可 inpaint），rtc 字段被安全忽略。
- **图像协议契约（易踩坑）**：`obs.images` 的值必须是 **CHW (3,224,224) uint8**
  且键名为 openpi 相机键（`base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb`）。
  服务端 `AlohaInputs._decode_aloha` 无条件执行 `einops.rearrange(img,"c h w -> h w c")`
  ——直接发 HWC 原图会被误转置成 `(640,3,480)` 并在 `ResizeImages` 的
  `PIL.Image.fromarray` 崩溃（报错 `(1, 1, 480), |u1`）。真实客户端
  （`pi05_adapter` websocket 路径）由 `runtime.build_observation` 负责
  `resize_with_pad 224 → CHW → openpi 键名` 映射，因此天然合规。
  端到端验证脚本 `recipes/pi0fast/test_pi0fast_client.py` 已按此契约构造合成观测
  （实测稳态推理 ~545ms，与训练共卡）。
- 客户端复用 `pi05` websocket 通道（`policy.type: pi05` + `pi05_backend: websocket`），
  仅 `lingbot_expected_model: pi0_fast` 校验握手；配置由
  `recipes/pi0fast/gen_client_config.py` 从 00092 客户端 YAML 派生：
  `infer/infer/config/lerobot/pi0fast_openpi_tongbot_16d_00036_ws_client.yaml`。
- norm_stats 安全契约同 PI05：只认 `<ckpt>/assets/norm_stats.json`（训练自动写入），
  identity/缺失直接报错拒绝推理。

## 产物

- 训练 ckpt：`<repo>/checkpoints/<exp>/checkpoints/<step>/pretrained_model/params` +
  `assets/norm_stats.json`（`checkpoints/last` 指向最新 step）
- 请求级日志：`recipes/pi0fast/pi0fast_requests.jsonl`
- 训练 dashboard：`<run>/train.log` + loss 曲线（`_train_log`）

## 附：已停服务

motus 推理（pid 3417279，占 42GB，2026-09-10 17:38 启动的验证残留）已停以让位训练。
恢复命令：

```bash
nohup /home/xk/miniconda3/envs/motus/bin/python -u /home/xk/data_Driven/recipes/motus/serve_tong_motus.py \
  --ckpt /home/xk/VLA/h100_ckpts/motus_tongbot_jiangsuan4/checkpoint_step_9000 \
  --port 1241 --denoise 10 \
  --instruction "Use both arms to complete the pick and place task." &
```
