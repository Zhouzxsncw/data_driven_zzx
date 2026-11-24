#!/usr/bin/env python3
"""向 cigai-agilex-train-openpi 的 src/openpi/training/config.py 注册
pi0_fast_tongbot / pi0_fast_tongbot_lora 两个 TrainConfig 预设。

幂等：检测到 "pi0_fast_tongbot" 已存在则直接退出。
用法: python patch_config_fast_tongbot.py [--repo /path/to/cigai-agilex-train-openpi]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARK = 'name="pi0_fast_tongbot"'
ANCHOR = '    TrainConfig(\n        name="pi05_libero",'
BACKUP_SUFFIX = ".orig_fast_tongbot"

# 插入的预设：
# - 权重走 openpi 官方 gs:// 路径；实际加载由 ~/.cache/openpi/openpi-assets/
#   checkpoints/pi0_fast_base 预置的本地目录/symlink 命中（maybe_download 本地优先）。
# - action_dim=16 对应 TongBot 左臂7|右臂7|左爪|右爪，无 PadStatesAndActions，必须严格 16。
# - action_horizon=15 + max_token_len=280：实测 FAST tokenize 16D×15 动作 + 长 prompt
#   ≈ 250 token（h=50 需 572 超限），280 留安全余量（官方双臂建议 ~250）。
SNIPPET = '''    TrainConfig(
        name="pi0_fast_tongbot",
        # TongBot 16D（左臂7 | 右臂7 | 左爪 | 右爪），PI0-FAST 数据管线无
        # PadStatesAndActions，action_dim 必须与数据严格一致。
        model=pi0_fast.Pi0FASTConfig(action_dim=16, action_horizon=15, max_token_len=280),
        data=LeRobotAlohaDataConfig(
            # repo_id / assets 由 YAML data.* 覆盖（指向各机器数据集目录）。
            assets=AssetsConfig(assets_dir=".", asset_id="."),
            default_prompt="null",
            # TongBot 双臂 16D 不走 Aloha14D 的 pi 空间翻转
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            # [左7关节 delta | 右7关节 delta | 左爪 abs | 右爪 abs]
            delta_action_dims=(7, 7, -1, -1),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.front",
                                "left_wrist_0_rgb": "observation.images.left",
                                "right_wrist_0_rgb": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        # 本地缓存命中：~/.cache/openpi/openpi-assets/checkpoints/pi0_fast_base/params
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    TrainConfig(
        # 低显存备选：LoRA 只训旁路，EMA 关闭
        name="pi0_fast_tongbot_lora",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=16, action_horizon=15, max_token_len=280, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(assets_dir=".", asset_id="."),
            default_prompt="null",
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            delta_action_dims=(7, 7, -1, -1),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.front",
                                "left_wrist_0_rgb": "observation.images.left",
                                "right_wrist_0_rgb": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=16, action_horizon=15, max_token_len=280, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
    ),
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="/home/xk/VLA/openpi/cigai-agilex-train-openpi",
        help="cigai-agilex-train-openpi 仓库根目录",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="直接指定 training/config.py 路径（如 vendored 推理端 "
        "infer/infer/policies/openpi/training/config.py），优先于 --repo",
    )
    args = parser.parse_args()

    if args.config_path:
        cfg_path = Path(args.config_path).expanduser()
    else:
        cfg_path = Path(args.repo) / "src" / "openpi" / "training" / "config.py"
    if not cfg_path.is_file():
        sys.exit(f"错误: 找不到 {cfg_path}")

    text = cfg_path.read_text(encoding="utf-8")
    if MARK in text:
        print(f"[skip] {cfg_path} 已包含 {MARK}，无需重复打补丁")
        return

    if ANCHOR not in text:
        sys.exit("错误: 找不到插入锚点（pi05_libero TrainConfig），config.py 结构可能已变化")

    backup = cfg_path.with_suffix(cfg_path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(cfg_path, backup)
        print(f"[backup] {cfg_path} -> {backup}")

    patched = text.replace(ANCHOR, SNIPPET + ANCHOR, 1)
    cfg_path.write_text(patched, encoding="utf-8")
    print(f"[ok] 已注册 pi0_fast_tongbot / pi0_fast_tongbot_lora -> {cfg_path}")


if __name__ == "__main__":
    main()
