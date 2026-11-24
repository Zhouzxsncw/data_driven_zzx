#!/usr/bin/env python3
"""
Motus-00091 开跑前的 CPU-only 静态校验（不碰 GPU，可与 DreamZero 训练并存）。

在 h100-103 上，Motus 的 8 卡 smoke 只能等 DreamZero-14B 让卡；但上次 smoke 挂在
数据链路（task_mode=multi 的 root/task_name 推导 → lerobot 回落到 hub → 该机不可达
hf.co 直接崩），这类错误根本不需要 GPU 就能提前暴露。本脚本把 GPU 之前会走的
全部路径先在 CPU 上走一遍：

  1) yaml 能解析 + 派生字段（action_chunk_size）+ 关键项断言
  2) 权重/数据集/T5 缓存/归一化表 文件级齐全
  3) LeRobotMotusDataset(multi) 真构造 + 随机采样若干条 + collate 成 batch
  4) 单条样本解码耗时 → 估算 8×8 worker 能否喂住 3~4 s/step

用法（远端，必须 cwd=MOTUS_ROOT）：
  CUDA_VISIBLE_DEVICES="" /data1/motus-env/bin/python /data1/motus/recipes/validate_motus_00091_h100.py \
      --config configs/tongbot_00091.yaml --samples 6
退出码 0 = 全部通过；非 0 = 有 FAIL（训练必然起不来，先别占卡）。
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

FAILS = []
WARNS = []


def check(cond, msg):
    if cond:
        print(f"  [ok]   {msg}")
    else:
        print(f"  [FAIL] {msg}")
        FAILS.append(msg)
    return cond


def warn(cond, msg):
    if not cond:
        print(f"  [WARN] {msg}")
        WARNS.append(msg)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tongbot_00091.yaml")
    ap.add_argument("--samples", type=int, default=6, help="随机采样条数（0=跳过数据链路）")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Motus 00091 静态校验  cwd={os.getcwd()}  python={sys.version.split()[0]}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print("=" * 70)

    # ---------- 1) 配置 ----------
    print("\n[1] 配置解析与断言")
    from omegaconf import OmegaConf

    cfg_path = Path(args.config)
    if not check(cfg_path.exists(), f"配置文件存在: {cfg_path}"):
        return 2
    cfg = OmegaConf.load(str(cfg_path))
    cfg.common.action_chunk_size = cfg.common.num_video_frames * cfg.common.video_action_freq_ratio
    print(f"      action_chunk_size={cfg.common.action_chunk_size} "
          f"(num_video_frames={cfg.common.num_video_frames} ratio={cfg.common.video_action_freq_ratio})")

    c = cfg.common
    check(c.action_dim == 16 and c.state_dim == 16, "16D（左7+右7+夹爪2）")
    check(c.global_downsample_rate == 3, "15fps → 5fps（每 3 帧取 1）")
    d = cfg.dataset
    check(d.type == "lerobot", f"dataset.type={d.type}")
    check(d.task_mode == "multi", f"task_mode={d.task_mode}")
    # 上次的致命坑：multi + root 直指数据集目录 → LeRobotDatasetMetadata 找不到 meta → hub → 崩
    root, task = str(d.params.root), str(d.task_name)
    check(os.path.isdir(os.path.join(root, task)),
          f"multi 模式 root 是父目录且 root/task_name 存在: {root}/{task}")
    check(os.path.isdir(os.path.join(root, task, "meta")),
          f"{root}/{task}/meta 存在（否则 lerobot 会回落 hub）")
    check(d.params.embodiment_type not in (None, "null"),
          f"embodiment_type={d.params.embodiment_type}")
    tr = cfg.training
    check(tr.batch_size >= 1, f"per-card batch_size={tr.batch_size}")
    print(f"      max_steps={tr.max_steps} lr={float(tr.learning_rate):.1e} warmup={tr.warmup_steps} "
          f"save_interval={cfg.system.save_interval} val_interval={cfg.system.val_interval} "
          f"num_workers={cfg.system.num_workers}")
    print(f"      checkpoint_dir(yaml)={cfg.system.checkpoint_dir}")
    warn(os.path.dirname(str(cfg.system.checkpoint_dir)).startswith("/data/"),
         "ckpt 落点不在 /data（/data1 只剩 ~366G，5B ckpt 单份就 60~80G）→ 启动器会 --checkpoint_dir 覆盖")

    # ---------- 2) 文件级齐全 ----------
    print("\n[2] 权重 / 数据 / 缓存 文件齐全性")
    m = cfg.model
    for name, p, min_bytes in [
        ("wan config.json", os.path.join(m.wan.config_path, "config.json"), 200),
        ("wan pretrain (model_states)", os.path.join(m.wan.checkpoint_path, "mp_rank_00_model_states.pt"), 1 << 30),
        ("Wan2.2 VAE", m.wan.vae_path, 1 << 30),
        ("VLM processor_config", os.path.join(m.vlm.checkpoint_path, "preprocessor_config.json"), 50),
        ("VLM tokenizer", os.path.join(m.vlm.checkpoint_path, "tokenizer_config.json"), 50),
        ("finetune 起点 mp_rank_00_model_states.pt",
         os.path.join(cfg.finetune.checkpoint_path, "mp_rank_00_model_states.pt"), 1 << 30),
    ]:
        check(os.path.isfile(p) and os.path.getsize(p) >= min_bytes, f"{name}: {p}")
    # umt5 fast tokenizer（缺 tokenizer.json 会退回慢速版，pad() 直接报 None）
    t5dir = os.path.join(str(d.params.t5_wan_path), "Wan2.2-TI2V-5B", "google", "umt5-xxl")
    warn(os.path.isdir(t5dir), f"umt5 目录存在: {t5dir}")

    ds_root = os.path.join(root, task)
    info = json.load(open(os.path.join(ds_root, "meta", "info.json")))
    n_ep, n_fr, fps = int(info["total_episodes"]), int(info["total_frames"]), float(info["fps"])
    check(info.get("codebase_version") == "v2.1", f"codebase_version={info.get('codebase_version')}（0.3.2 只认 v2.1）")
    print(f"      episodes={n_ep} frames={n_fr} fps={fps} → 约 {int(n_fr // c.global_downsample_rate)} 个可用采样点")
    feats = info.get("features", {})
    check(list(feats.get("observation.state", {}).get("shape", [])) == [16], "observation.state shape=[16]")
    check(list(feats.get("action", {}).get("shape", [])) == [16], "action shape=[16]")
    cams = [k for k in feats if k.startswith("observation.images")]
    check(len(cams) == 3, f"3 路相机: {cams}")
    try:
        from data.lerobot.camera_views import resolve_image_source, require_three_cam_features
        require_three_cam_features(feats)
        mode, keys = resolve_image_source(feats)
        check(mode in ("three_cam", "concatenated"),
              f"loader 必须吃三路（three_cam/concatenated），实际 {mode} keys={list(keys)}")
    except Exception as e:
        print(f"  [FAIL] 三路相机解析: {e}")
        FAILS.append(f"三路相机解析: {e}")

    t5_files = sorted(Path(ds_root).glob("t5_embedding/*.pt"))
    check(len(t5_files) == n_ep, f"T5 episode 级缓存 {len(t5_files)}/{n_ep}（缺则训练期会现编码 T5，抢 CPU）")
    eps = [json.loads(l) for l in open(os.path.join(ds_root, "meta", "episodes.jsonl"))]
    have_ptr = sum(1 for e in eps if e.get("t5_embedding_path"))
    check(have_ptr == n_ep, f"episodes.jsonl 里 t5_embedding_path 字段 {have_ptr}/{n_ep}")
    missing_pt = sum(1 for e in eps
                     if e.get("t5_embedding_path") and not os.path.isfile(os.path.join(ds_root, e["t5_embedding_path"])))
    check(missing_pt == 0, f"t5_embedding_path 指向的 .pt 全部存在（缺 {missing_pt}）")

    stat_file = Path("data/utils/stat.json")
    if check(stat_file.is_file(), f"归一化表 {stat_file}"):
        stat = json.load(open(stat_file))
        emb = str(d.params.embodiment_type)
        if check(emb in stat, f"stat.json 含 embodiment '{emb}'"):
            e = stat[emb]
            check(len(e["min"]) == 16 and len(e["max"]) == 16, f"{emb}: min/max 均 16 维")
            rng = [abs(a - b) for a, b in zip(e["min"], e["max"])]
            check(max(rng) > 1e-3, f"{emb}: 各维 min<max（最小跨度 {min(rng):.4f}，若≈0 说明该维没激励）")
            print(f"        dim-range: {[round(x, 2) for x in rng]}")
            base = stat.get("tong_dual_arm")
            if base:
                warn(max(abs(a - b_) for a, b_ in zip(e["min"] + e["max"], base["min"] + base["max"])) < 0.5,
                     "与通用 tong_dual_arm 统计量不至于差一个量级（差太大说明 norm 取错数据集）")

    # ---------- 3) 数据链路（真构造 dataset + 采样） ----------
    if args.samples > 0:
        print(f"\n[3] LeRobotMotusDataset 构造 + 采样 {args.samples} 条（这一步就是上次 smoke 炸的地方）")
        t0 = time.time()
        try:
            from data.dataset import create_dataset, collate_fn
        except Exception:
            traceback.print_exc()
            FAILS.append("import data.dataset 失败")
            return 2
        try:
            ds = create_dataset(cfg, val=False)
        except Exception:
            traceback.print_exc()
            FAILS.append("create_dataset(val=False) 抛异常 —— 训练必然起不来")
            ds = None
        if ds is not None:
            print(f"      构造耗时 {time.time() - t0:.1f}s，len={len(ds)}")
            check(len(ds) > 1000, f"数据集长度 {len(ds)} 合理（>1000）")
            check(getattr(ds, "action_min", None) is not None and len(ds.action_min) == 16,
                  "ds.action_min 16 维（归一化表已挂上）")
            check(bool(getattr(ds, "has_three_cam", False) or getattr(ds, "has_concat", False)),
                  f"dataset Image source 必须是 three_cam（TongBot front/left/right T 拼图），"
                  f"实际 mode={getattr(ds, 'image_mode', None)} keys={getattr(ds, 'image_keys', None)}")
            idxs = random.Random(0).sample(range(len(ds)), args.samples)
            times, ok = [], 0
            for i in idxs:
                try:
                    s = ds[int(i)]
                    if s is None:
                        print(f"      idx={i} 返回 None（采样窗口越界，偶发可接受）")
                        continue
                    assert s["video_frames"].shape[0] == c.num_video_frames, s["video_frames"].shape
                    hh, ww = s["video_frames"].shape[-2:]
                    check_ok = (hh, ww) == (c.video_height, c.video_width)
                    assert check_ok, (hh, ww)
                    assert s["action_sequence"].shape == (c.action_chunk_size, c.action_dim), s["action_sequence"].shape
                    assert s["first_frame"].shape[0] == 3, s["first_frame"].shape
                    ok += 1
                    if ok == 1:
                        print(f"      样本形状  video={tuple(s['video_frames'].shape)} "
                              f"action={tuple(s['action_sequence'].shape)} "
                              f"state={tuple(s['initial_state'].shape)} "
                              f"lang={tuple(s['language_embedding'].shape) if s.get('language_embedding') is not None else None} "
                              f"vlm={type(s.get('vlm_inputs')).__name__}")
                    amax = float(s["action_sequence"].abs().max())
                    warn(amax < 20.0, f"归一化后 action 幅度 {amax:.2f} 合理（>20 说明 norm 没生效）")
                except Exception:
                    traceback.print_exc()
                    FAILS.append(f"idx={i} 采样失败")
                times.append(time.time() - t0)
            if ok:
                per = [times[k + 1] - times[k] for k in range(len(times) - 1)] or [times[0]]
                med = statistics.median(per)
                # 每 step 每 rank 要 bs 个样本；一个 worker 出 1 个样本要 med 秒，
                # num_workers 并发 → 每 rank 每 step 理论下限 med*bs/num_workers
                est = med * tr.batch_size / max(1, cfg.system.num_workers)
                print(f"      解码单条中位数 {med:.2f}s（pyav 3 路视频）→ "
                      f"{cfg.system.num_workers} worker 下每 rank 每 step 下限约 {est:.2f}s")
                warn(est < 3.0,
                     f"DataLoader 理论下限 {est:.2f}s/step 已接近或超过 GPU 步时（3~4s），"
                     f"建议把 system.num_workers 提到 {int(max(1, -(-med * tr.batch_size // 3)))} 以上")
            else:
                FAILS.append("没有任何一条样本成功")

            print("\n[3b] collate_fn 组 batch")
            try:
                batch = [ds[int(i)] for i in idxs[:4]]
                batch = [b for b in batch if b is not None]
                out = collate_fn(batch)
                check(out is not None, "collate 返回非 None")
                if out:
                    print(f"      batch: video={tuple(out['video_frames'].shape)} "
                          f"action={tuple(out['action_sequence'].shape)} "
                          f"state={tuple(out['initial_state'].shape) if out.get('initial_state') is not None else None} "
                          f"lang={tuple(out['language_embedding'].shape) if out.get('language_embedding') is not None else None}")
                    check(out["language_embedding"] is not None, "language_embedding 非 None（T5 缓存通路 OK）")
            except Exception:
                traceback.print_exc()
                FAILS.append("collate_fn 抛异常")

    print("\n" + "=" * 70)
    if WARNS:
        print(f"WARN {len(WARNS)} 条：")
        for w in WARNS:
            print(f"  - {w}")
    if FAILS:
        print(f"FAIL {len(FAILS)} 条 —— 不要占卡开跑，先修：")
        for f in FAILS:
            print(f"  ✗ {f}")
        return 1
    print("ALL CHECKS PASSED —— 数据/权重/配置链路就绪，剩下的只有 GPU 侧风险（显存/compile/NCCL）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
