# 原生 ALOHA 推理环境（TongBot）

本仓库 `infer/config/aloha/` 专供 **TongBot** 真机推理。状态维固定为：

- `arm_joint_state_dim: **8**`（每臂 7 关节 + 1 夹爪）
- 左臂：`arm_joint_state_names: [left]` → 模型 8 维，`keep_dims: [0..6, 14]`
- 双臂：`names: [left, right]` → 模型 16 维，`keep_dims` 做 L8|R8 ↔ TongBot 重排

训练该 TongBot ckpt 时请用同样的 dim=8；不要用 agilex 仓库里默认的 dim=7 YAML 直接推理。

---

## 模型与依赖一览

| 模型 | `policy.type` | 配置目录 | Python 包（在推理环境内） |
|------|---------------|----------|---------------------------|
| ACT | `act` + `act_backend: aloha` | `aloha/act_aloha_*` | `torch`、`opencv-python`、`einops`、`numpy`、`pyyaml` |
| DP | `aloha_dp` | `aloha/dp_aloha_*` | 上表全部 + **`robomimic`** + **`diffusers`** |
| RDT | `aloha_rdt` | `aloha/rdt_aloha_*` | ACT 依赖 + **`diffusers`** + **`timm`**（一般不需要 robomimic） |

权重与统计（三家相同目录约定）：

```text
.../pretrained_model/   # 或你指向的 ckpt 目录
  policy_best.ckpt
  dataset_stats.pkl
```

- ACT / RDT：`policy_best.ckpt` 为 **flat** `state_dict`
- DP：`policy_best.ckpt` 为 **`{nets, ema}`**

---

## 推荐：单独 conda 环境 `aloha-infer`

与 Pi0.5 的 `openpi-pi05-infer` / `lerobot` 环境分开，避免依赖互相踩。

```bash
# 1. 创建环境（Python 3.10 与训练侧接近即可）
conda create -n aloha-infer python=3.10 -y
conda activate aloha-infer

# 2. PyTorch（按本机 CUDA 选官方命令；示例 CUDA 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. 公共依赖（ACT / DP / RDT 都需要）
pip install opencv-python einops numpy pyyaml h5py packaging

# 4. RDT 额外（ACT 不需要；DP 也可装）
pip install timm diffusers==0.35.1 huggingface-hub

# 5. DP 额外 — 与训练仓一致（cigai-agilex-train 根目录已 vendored robomimic，
#    含 algo/diffusion_policy.py）。PyPI 0.3.x 没有该模块，不要 pip install robomimic。
pip uninstall -y robomimic
# 优先（与 ckpt 同源）：把训练仓根目录装进环境（--no-deps 避开 egl_probe）
pip install -e /path/to/cigai-agilex-train --no-deps
# 若暂时没有训练仓：退而求其次用官方分支
# pip install "git+https://github.com/ARISE-Initiative/robomimic.git@diffusion-policy-mg" --no-deps
pip install h5py  # 若尚未安装；diffusers 训练侧为 0.35.1
```

快速自检：

```bash
conda activate aloha-infer
python - <<'PY'
import torch, cv2, einops, yaml
print("ACT base OK", torch.__version__)
try:
    import timm
    print("timm OK", timm.__version__, "(RDT 需要)")
except ImportError as e:
    print("timm MISSING (仅 RDT 需要):", e)
try:
    import diffusers
    print("diffusers OK", diffusers.__version__)
except ImportError as e:
    print("diffusers MISSING (RDT/DP 需要):", e)
try:
    from robomimic.algo.diffusion_policy import ConditionalUnet1D
    from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
    print("robomimic DP OK")
except ImportError as e:
    print("robomimic MISSING/WRONG VERSION (仅 DP 需要 diffusion-policy-mg):", e)
PY
```

---

## 启动推理（TongBot）

在仓库根目录：

```bash
conda activate aloha-infer
cd /path/to/cigai-vla-leroboot
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 改对应 YAML 里的 policy.ckpt_dir 指向含 policy_best.ckpt + dataset_stats.pkl 的目录

# ACT 左臂
python3 infer/cli/run_batch_infer.py --config infer/config/aloha/act_aloha_left_arm_direct.yaml

# DP 左臂
python3 infer/cli/run_batch_infer.py --config infer/config/aloha/dp_aloha_left_arm_direct.yaml

# RDT 双臂
python3 infer/cli/run_batch_infer.py --config infer/config/aloha/rdt_aloha_dual_arm_gripper_direct.yaml
```

硬件侧与 Pi0.5 相同：下位机夹爪（`lower_machine`）、相机 serial、`local_env.yaml` / YAML 里 IP。见 [../README.md](../README.md)。

---

## 常见报错

| 报错 | 处理 |
|------|------|
| `No module named 'timm'` | 仅 RDT：`pip install timm`（ACT 路径已懒加载，不需要） |
| `No module named 'diffusers'` | RDT/DP：`pip install diffusers` |
| `No module named 'robomimic'` / `robomimic.algo.diffusion_policy` | 仅 DP：装训练仓 vendored robomimic（`pip install -e /path/to/cigai-agilex-train --no-deps`），不要用 PyPI 0.3.x |
| `EMAModel.__init__() missing ... 'parameters'` | DP：新版 `diffusers` API 变化；已用 `_build_ema` 兼容。若仍报错，确认推理代码已更新到含该 shim 的版本 |
| `ALOHA_ARM_DIM` / dim≠8 | TongBot 必须 `arm_joint_state_dim: 8` |
| `ALOHA_KEEP_DIMS_LAYOUT` | 左臂用 `[0..6,14]`；双臂用 L8\|R8↔TongBot 重排 |
| checkpoint 含 `nets` / ACT 报 Diffusion 格式 | DP 权重误用于 ACT/RDT，或反过来；换对 `policy.type` |
| `action_dim` 与 `keep_dims` 不一致 | 确认 YAML `arm_joint_state_dim: 8` 且与训练 TongBot ckpt 一致 |
| `.cuda()` / 需要 CUDA | 原生 ALOHA policy 构建时强制 CUDA，需有 GPU 的机器 |

---

## 相关文档

- 配置索引：[../config/README.md](../config/README.md)
- ACT adapter：[../policies/act_adapter.md](../policies/act_adapter.md)
- DP adapter：[../policies/aloha_dp_adapter.md](../policies/aloha_dp_adapter.md)
- RDT adapter：[../policies/aloha_rdt_adapter.md](../policies/aloha_rdt_adapter.md)
- 批量手册：[BATCH_INFER.md](BATCH_INFER.md)
