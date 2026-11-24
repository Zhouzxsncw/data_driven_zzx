#!/usr/bin/env python3
"""离线校验 DreamZero SFT 训练 YAML（hydra compose 干跑，不需要 GPU）。

在 H100 上或本机都能跑：只依赖 hydra/omegaconf + RLinf 的 config 目录。
它做三件事：
  1) 让 hydra 真的合并一遍 defaults（能抓出 struct 键、插值、缺失 group 等错误）；
  2) 断言我们最在意的几个键（分辨率 ↔ frame_seqlen、cosine 步数联动、
     FSDP bf16、组件权重路径非空、dataset/metadata 已填）；
  3) 打印一份精简后的有效配置，方便和 5B 对比。

用法:
  python check_sft_config.py                       # 默认查 14B
  python check_sft_config.py --config-name tong00073_sft_dreamzero_5b
  python check_sft_config.py --set runner.max_steps=6000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-dir", default=os.environ.get("SFT_CONFIG_DIR"),
                   help="RLinf examples/sft/config 目录（默认自动探测 ~/VLA/RLinf 与 /data1/RLinf）")
    p.add_argument("--config-name", default="tong00073_sft_dreamzero_14b")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="额外 hydra CLI 覆盖，可重复；例 --set actor.micro_batch_size=2")
    p.add_argument("--allow-missing-paths", action="store_true",
                   help="不检查权重/数据集文件是否在本机存在（本机干跑时用）")
    p.add_argument("--full", action="store_true", help="打印完整合并后的配置")
    return p.parse_args()


def find_config_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    for cand in (
        Path.home() / "VLA/RLinf/examples/sft/config",
        Path("/data1/RLinf/examples/sft/config"),
        here_probe(),
    ):
        if cand and Path(cand).is_dir():
            return Path(cand).resolve()
    sys.exit("找不到 RLinf examples/sft/config，请给 --config-dir")


def here_probe() -> Path:
    # 仓库内相对探测：.../recipes/dreamzero -> 同级 VLA/RLinf
    return Path(__file__).resolve().parent.parent.parent.parent / "VLA/RLinf/examples/sft/config"


def frame_seqlen_of(h: int, w: int, vae_down: int, patch: int = 2) -> int:
    """VAE 空间下采样 + DiT patch (1,2,2) -> 每帧 latent token 数。

    Wan2.1_VAE   8x  -> 176x320: 22x40 -> 11x20  = 220   （14B tong）
    Wan2.2_VAE38 16x -> 176x320: 11x20 -> 5x10   = 50    （5B）
    352x640 + 8x VAE          -> 22x40 = 880             （RLinf 14B 默认）

    不要乘视图数：tong_dual_arm 虽然把三路相机沿 H 竖排成 528x320
    （data_transforms/tong_dual_arm.py:concat_multiview_video），但 action head
    会先 interpolate 回 (target_video_height, target_video_width) 才进 VAE
    （wan_flow_matching_action_tf.py L648-657），所以仍是 176x320 → 220。
    曾按 3 视图推成 660，结果 step 0 报 Sequence length does not match block layout。
    """
    return (h // vae_down // patch) * (w // vae_down // patch)


def vae_downsample_of(cfg: DictConfig) -> int:
    """从 vae_cfg._target_ 推 VAE 空间压缩倍率（VAE38 = 16x16，Wan2.1 VAE = 8x8）。"""
    tgt = str(cfg.actor.model.action_head_cfg.config.get("vae_cfg", {}).get("_target_", ""))
    return 16 if "VAE38" in tgt or "WanVideoVAE38" in tgt else 8


def check(cfg: DictConfig, args: argparse.Namespace) -> list[str]:
    errs: list[str] = []
    m = cfg.actor.model
    ahc = m.action_head_cfg.config
    dmc = ahc.diffusion_model_cfg

    # --- 1) 分辨率 ↔ frame_seqlen 必须自洽 ---
    # 两边都要管（2026-09-10 读代码+实跑核实的真实语义）：
    #   - 训练 teacher-forcing 路径用 self.frame_seqlen 算 noisy_frames / num_image_blocks
    #     并列校 block 布局（wan_video_dit_action_casual_chunk.py L884-900），写错直接 step 0 报
    #     ValueError: Sequence length does not match block layout；
    #   - 推理 kv_cache 路径用 max_attention_size = local_attn_size * frame_seqlen 截 KV（L1072-1073）。
    # 所以 frame_seqlen 必须等于“进 VAE 的合成帧”的 latent token/帧（见 frame_seqlen_of 说明）。
    h, w = int(m.target_video_height), int(m.target_video_width)
    down = vae_downsample_of(cfg)
    expect = frame_seqlen_of(h, w, down)
    got = int(dmc.frame_seqlen)
    z_dim = int(getattr(dmc, "out_dim", 0) or 0)
    print(f"[cfg] target_video   = {h}x{w}（进 VAE 的合成帧）  VAE {down}x + patch2 -> frame_seqlen 应为 {expect}，实际 {got} (out_dim={z_dim})")
    if got != expect:
        errs.append(f"frame_seqlen={got} 与合成帧 {h}x{w} / VAE{down}x 推导值 {expect} 不符")

    # 局部注意力窗口（推理侧显存/语义关键量）
    local_frames = int(dmc.max_chunk_size) * int(dmc.num_frame_per_block) + 1
    print(f"[cfg] local_attn     = {local_frames} 帧 = {local_frames * got} token（KV 上限，不随 episode 增长）")

    # --- 2) cosine 调度与 max_steps 联动 ---
    ms, tts = int(cfg.runner.max_steps), int(cfg.actor.optim.total_training_steps)
    print(f"[cfg] max_steps      = {ms}, total_training_steps = {tts}")
    if ms > 0 and tts != ms:
        errs.append(f"optim.total_training_steps({tts}) != runner.max_steps({ms})")

    # --- 3) 批量与卡数 ---
    mbs, gbs = int(cfg.actor.micro_batch_size), int(cfg.actor.global_batch_size)
    print(f"[cfg] batch          = micro {mbs} x 8 卡 = {mbs * 8}，global_batch_size {gbs}")
    if gbs % mbs:
        errs.append(f"global_batch_size({gbs}) 不是 micro_batch_size({mbs}) 的整数倍")

    # --- 4) FSDP 精度/分片 ---
    fs = cfg.actor.fsdp_config
    print(f"[cfg] fsdp           = {fs.strategy}/{fs.sharding_strategy} mp={fs.mixed_precision.param_dtype}"
          f" gc={fs.gradient_checkpointing} offload={cfg.actor.get('enable_offload', 'n/a')}")
    if str(fs.mixed_precision.param_dtype).lower() != "bf16":
        errs.append("mixed_precision.param_dtype 必须是 bf16，否则 14B 装不下")
    if str(fs.sharding_strategy).lower() not in ("full_shard", "shard_grad_op"):
        errs.append(f"sharding_strategy={fs.sharding_strategy} 不适合 8x80G 全参 14B")

    # --- 5) 组件权重与数据（键必须存在且非空）---
    keys = ["tokenizer_path", "diffusion_model_pretrained_path", "image_encoder_pretrained_path",
            "text_encoder_pretrained_path", "vae_pretrained_path", "metadata_json_path"]
    for k in keys:
        v = m.get(k, None)
        if v is None or str(v).startswith("null"):
            errs.append(f"actor.model.{k} 未设置")
            continue
        print(f"[cfg] {k:33s}= {v}")
    print(f"[cfg] dataset                          = {cfg.data.train_data_paths}")

    # --- 6) 结构正确性：14B 必须走 Wan2.1 VAE + i2v 骨干 ---
    if "14b" in args.config_name:
        if str(dmc.model_type).lower() != "i2v":
            errs.append(f"14B 骨干 model_type 应为 i2v，实际 {dmc.model_type}")
        if "Wan2.1_VAE" not in str(m.vae_pretrained_path):
            errs.append(f"14B 的 vae 应为 Wan2.1_VAE.pth，实际 {m.vae_pretrained_path}")
    ids = [int(i) for i in ahc.action_loss_embodiment_ids]
    print(f"[cfg] action_loss_embodiment_ids = {ids}, action_horizon = {m.action_horizon}, "
          f"num_action_per_block = {m.num_action_per_block}, embodiment = {m.get('embodiment_tag')}")
    if int(m.action_horizon) != int(m.num_action_per_block):
        errs.append("action_horizon 应等于 num_action_per_block（单 block 出 16 个动作）")

    if not args.allow_missing_paths:
        for k in keys + ["train_data_paths"]:
            v = str(m.get(k) if k != "train_data_paths" else cfg.data.train_data_paths)
            if not Path(v).exists():
                errs.append(f"{k} 指向的路径在本机不存在: {v}")

    if args.full:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    return errs


def main() -> int:
    args = parse_args()
    cdir = find_config_dir(args.config_dir)
    os.environ.setdefault("EMBODIED_PATH", str(cdir.parent))  # hydra searchpath 用
    print(f"[cfg] config-dir  = {cdir}")
    print(f"[cfg] hydra       = {hydra.__version__}")
    print(f"[cfg] config-name = {args.config_name}")

    from hydra import compose, initialize_config_dir
    try:
        with initialize_config_dir(version_base=None, config_dir=str(cdir)):
            cfg = compose(config_name=args.config_name, overrides=args.overrides)
    except Exception as e:  # noqa: BLE001
        print(f"\nFAIL compose: {type(e).__name__}: {e}")
        return 2

    errs = check(cfg, args)
    print()
    for e in errs:
        print(f"  ✗ {e}")
    if errs:
        print(f"\nFAIL {len(errs)} 项")
        return 1
    print("OK compose + 断言全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
