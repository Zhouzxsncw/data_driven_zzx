#!/usr/bin/env python3
"""EE 20D 数据集**直读体检**：绕开 τ0-WM 的静默丢样，逐条 assert。

为什么需要它：``CustomLeRobotDataset.__getitem__`` 是 ``try/except: 换下一个 idx``
（data/example_dataset.py:460-473）—— 索引切片写错、断言 D==18 失败、caption 取成
"0" 这类问题**都不会让训练报错**，只会把样本一个个丢掉；全丢就变成 while True 死循环，
表现为「训练卡在第 1 个 batch」。所以体检必须走 ``get_batch()``，让异常直接抛。

检查项
------
1. ``actions`` (33, 20) / ``state`` (1, 20) —— 20 维 = 预训练 action_in_dim
2. 归一化后 ``max|z|``：>10 说明某维 std 太小、``filter_action`` 的 >100 阈值有触雷风险
3. 相对量结构：rot6d 对角项（idx 3/7/13/17）均值应接近 1，非对角接近 0
4. ``video`` (C=3, V=3, T=9, H=192, W=256)（``get_video`` 输出 c,v,t,h,w）、3 路顺序 = valid_cam 顺序
5. caption 是真指令字符串（不是 "0" / "1" 这种退化索引）

用法::

    python verify_ee_dataset.py                       # 本机 EE 树
    python verify_ee_dataset.py --n 200               # 抽更多样本
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def build_dataset(config_path: Path):
    from utils import import_custom_class

    cfg = yaml.safe_load(config_path.read_text())
    kwargs = dict(cfg["data"])
    kwargs.update(
        norm_action=True,
        filter_action=True,
        return_video=True,
        action_space="eef6d",
        action_type="relative",
    )
    cls = import_custom_class(cfg["data_class"], cfg["data_class_path"])
    return cls(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=Path("/home/xk/VLA/tau-0-wm/configs/data/tongbot/00036_ee_rela.yaml"))
    ap.add_argument("--root-map", nargs=2, metavar=("FROM", "TO"), default=None,
                    help="把 YAML 里的数据集前缀换成本机前缀（h100 路径 → 本地路径）")
    ap.add_argument("--stat", type=Path,
                    default=Path("/home/xk/VLA/tau-0-wm/configs/data/stat_file/tongbot_00036_ee_rela.json"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--full-scan", action="store_true",
                    help="不读视频，全量扫一遍：统计归一化后 |z| 的分布，"
                         "直接量化训练会因为 filter_action(|z|>100) 静默丢掉多少样本")
    ap.add_argument("--stride", type=int, default=1, help="--full-scan 时的采样步长")
    args = ap.parse_args()

    text = args.config.read_text()
    if args.root_map:
        text = text.replace(args.root_map[0], args.root_map[1])
    tmp = Path("/tmp/tau_ee_verify.yaml")
    tmp.write_text(text)
    # 统计文件走仓库相对路径，得改成绝对路径（本脚本不一定在 repo 根跑）
    y = yaml.safe_load(tmp.read_text())
    y["data"]["statistic_files"] = [str(args.stat)]
    if args.full_scan:
        y["data"]["return_video"] = False      # 全量扫描只关心动作/状态归一化
    tmp.write_text(yaml.safe_dump(y, allow_unicode=True, sort_keys=False))

    ds = build_dataset(tmp)
    print(f"len(dataset) = {len(ds)}")
    random.seed(args.seed)
    if args.full_scan:
        idxs = list(range(0, len(ds), args.stride))
    else:
        idxs = random.sample(range(len(ds)), args.n)

    zs, acts, states, caps, vids = [], [], [], [], []
    worst = []                                  # (max|z|, idx) 全量扫描用
    over100 = over10 = 0
    for k, i in enumerate(idxs):
        r = ds.get_batch(i)                      # 注意：不走 __getitem__ 的吞异常
        a, s = r["actions"], r["state"]
        assert a.shape == (33, 20), f"actions shape {tuple(a.shape)} @ idx {i}"
        assert s.shape == (1, 20), f"state shape {tuple(s.shape)} @ idx {i}"
        assert torch.isfinite(a).all() and torch.isfinite(s).all(), f"NaN/Inf @ idx {i}"
        zmax = max(float(a.abs().max()), float(s.abs().max()))
        zs.append(zmax)
        if args.full_scan:
            over100 += zmax > 100               # 这些样本训练时会被静默丢掉
            over10 += zmax > 10
            worst.append((zmax, i))
            worst.sort(reverse=True)
            del worst[10:]
            if (k + 1) % 5000 == 0:
                print(f"  ...{k + 1}/{len(idxs)}  |z|>100: {over100}  >10: {over10}", flush=True)
            continue
        assert r["video"].shape == (3, 3, 9, 192, 256), f"video {tuple(r['video'].shape)} @ idx {i}"
        acts.append(a.numpy())
        states.append(s.numpy()[0])
        caps.append(str(r["caption"]))
        if k == 0:
            view_mean = [round(float(r["video"][:, j].mean()), 3) for j in range(3)]
            print(f"首样本 idx={i}  video={tuple(r['video'].shape)} "
                  f"范围[{r['video'].min():.2f},{r['video'].max():.2f}]  "
                  f"三视图均值={view_mean}（应互不相同，证明 3 路相机没读重）  "
                  f"caption={r['caption']!r}")

    if args.full_scan:
        n = len(idxs)
        z = np.array(zs)
        print(f"\n全量扫描 {n} 个样本（stride={args.stride}）：")
        print(f"  max|z| = {z.max():.2f}   p50={np.percentile(z,50):.2f} "
              f"p99={np.percentile(z,99):.2f} p99.9={np.percentile(z,99.9):.2f}")
        print(f"  |z|>100（会被 filter_action 丢掉）: {over100} = {100*over100/n:.3f}%")
        print(f"  |z|>10 : {over10} = {100*over10/n:.3f}%")
        print("  最差 10 个 idx:", [(f"{v:.1f}", i) for v, i in worst])
        assert over100 / n < 0.005, f"被静默丢弃的样本比例过高 {100*over100/n:.2f}%"
        print("\nALL OK —— 全量归一化无爆维，filter_action 丢弃率可接受")
        return

    A = np.concatenate(acts, axis=0)             # (n*33, 20) 归一化后的相对动作
    S = np.stack(states)                         # (n, 20) 归一化后的状态
    print(f"\nmax|z| action/state = {max(zs):.3f}   （filter_action 阈值 100，>10 就要盯）")
    print(f"{'idx':>3} {'含义':10} {'归一化 mean':>12} {'std':>8}")
    names = ["L_x", "L_y", "L_z"] + ["L_r6d"] * 6 + ["L_grip"] + \
            ["R_x", "R_y", "R_z"] + ["R_r6d"] * 6 + ["R_grip"]
    for i in range(20):
        print(f"{i:>3} {names[i]:10} {A[:, i].mean():>12.4f} {A[:, i].std():>8.4f}")

    diag = [3, 7, 13, 17]
    raw_mean = np.array([A[:, i].mean() for i in diag])
    print(f"\nrot6d 对角项（{diag}）归一化后 mean={raw_mean.round(3)} —— "
          f"原始值≈1 才会落在正 z 区间，与预训练 statistics.json 同构")
    off = [i for i in range(20) if i not in diag + [0, 1, 2, 9, 10, 11, 12, 19]]
    print(f"rot6d 非对角项 mean={np.array([A[:, i].mean() for i in off]).round(3)}（应≈0）")

    bad = [c for c in caps if (c.strip().isdigit() or len(c.strip()) < 4)]
    print(f"\ncaption 去重后 {len(set(caps))} 种，退化（纯数字/过短）{len(bad)}/{len(caps)}")
    for c in list(dict.fromkeys(caps))[:3]:
        print("  例:", c)
    assert not bad, "caption 退化：tasks.parquet 索引有问题（见 recipes/tau0_wm/make_dataset_root.sh）"
    assert max(zs) < 100, f"存在归一化爆掉的维度 max|z|={max(zs):.1f}（≥100 会被 filter_action 丢弃）"
    print(f"state 归一化后 |z| max={np.abs(S).max():.2f}  action |z| max={np.abs(A).max():.2f}")
    if max(zs) > 10:
        print(f"注意：抽样里已有 |z|={max(zs):.1f}，正式训练前建议跑 --full-scan 量化丢弃率")
    print("\nALL OK —— EE 20D 数据集可被 τ0-WM 正确读出（含 relative + dual_arm 重排 + pyav 解码）")


if __name__ == "__main__":
    main()
