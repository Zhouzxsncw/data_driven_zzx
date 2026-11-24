# 通用工具 (tools/)

推理、数据采集与训练共用的工具，按功能分组。

## 目录结构

```
tools/
├── policy/       # 策略推理共用（帧构造、batch、observation keys、dataset stats）
├── camera/       # 相机分辨率、RealSense profile、checkpoint 校验
├── dataset/      # LeRobot 数据集准备、修复、验证、离线打包
├── bootstrap/    # 训练/评估环境 shim（VideoReader、HF cache、Pi0.5 环境验证）
├── offline/      # 离线 Pi0.5 推理回放、train vs infer 观测对比
└── sync/         # 时钟校准、训练工作树 rsync 部署
```

## 推理侧常用

```bash
# checkpoint 相机分辨率
python3 -m tools.camera.query_camera_resolutions --checkpoint /path/to/ckpt

# 离线对比真机 vs 训练观测
python3 tools/offline/compare_real_vs_train_obs.py ...
```

Python 导入示例：

```python
from tools.policy.policy_frame_utils import camera_image_to_model_tensor
from tools.camera.checkpoint_camera_resolution import resolve_checkpoint_image_hw
from tools.policy.lerobot_dataset_stats import load_local_dataset_stats
```

## 训练侧常用（维护者）

```bash
bash tools/dataset/bundle_offline_data.sh
bash tools/bootstrap/verify_pi05_env.sh
python3 tools/dataset/ensure_dataset_quantile_stats.py --root data/lerobot/my_dataset

# 部署到训练机（必须同时同步 train/ + tools/，否则会 No module named 'tools'）
bash tools/sync/rsync_train_workspace.sh h100-103 /home/zgyd/new_lerobot_train
```

`tools/bootstrap/` 是训练入口强依赖：`train/tools/run_lerobot_train.py` 会 import 它。  
训练专用脚本（曲线导出、checkpoint 批量 eval 等）在 `train/tools/`。

## 相关文档

| 文档 | 内容 |
|------|------|
| [../infer/README.md](../infer/README.md) | 真机推理 |
| [../infer/docs/BATCH_INFER.md](../infer/docs/BATCH_INFER.md) | 批量消融 |
| [../../docs/FORKS.md](../../docs/FORKS.md) | 训练 fork |
| [../vla_eval/README.md](../vla_eval/README.md) | VLA 自动评估 |
