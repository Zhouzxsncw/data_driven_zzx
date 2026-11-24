#!/usr/bin/env python3
"""给 FastWAM trainer 打「dataloader 多进程上下文可切换」补丁（幂等，带 --check，自动备份）。

为什么需要（2026-09-14 resume 三次 global OOM 的实测结论）：
  trainer 用 ZeRO-1 + CPU offload，优化器状态 72G 常驻在**主进程**匿名内存里（主进程 RssAnon
  约 105-127G）。DataLoader 默认 fork 出 num_workers 个子进程，子进程初始与父进程 100% 共享页
  （/data/.../logs/mem_probe.log 里 worker_rss_sum=214G / 2 个 worker，即每个 RSS≈107G≈父进程），
  但 worker 一旦写脏继承来的页就触发 COW 复制：mem_probe 采到 machine_anon 在 worker 起来后
  40s 内从 117G → 150G → 180G（基线空闲只有 4G），于是 188G 物理内存 + 64G swap 被一起吃光。
  worker 数从 4 降到 2 并没有救活（只是把越界时间推迟），因为脏页速率才是主因。

修法：
  把 DataLoader 的 multiprocessing_context 做成可配置（env FASTWAM_DLOADER_MP）：
    spawn      —— 子进程不继承父进程地址空间，只收 1.3MiB 的数据集 pickle（已实测 RobotVideoDataset
                  可 pickle 且构建仅 7.8s），彻底消除 COW 脏页放大；代价是每个 worker 多 1-2s 启动。
    fork/未设   —— 保持原行为。
  取数顺序仍由主进程里的 ResumableEpochSampler 决定，换上下文不改变样本顺序，不影响续训可复现性。

用法：
  python patch_dataloader_mp_ctx.py --check    # 未打退出码 1
  python patch_dataloader_mp_ctx.py            # 打补丁
  python patch_dataloader_mp_ctx.py --revert   # 还原
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TRAINER = Path("/home/xk/VLA/FastWAM/src/fastwam/trainer.py")
BACKUP = TRAINER.with_name("trainer.py.orig_pre_mp_ctx")
MARK = 'multiprocessing_context=(os.environ.get("FASTWAM_DLOADER_MP") or None)'

ANCHOR = """            sampler=self.train_sampler,
            num_workers=self.num_workers,
"""
NEW = (
    "            sampler=self.train_sampler,\n"
    "            num_workers=self.num_workers,\n"
    "            # num_workers>0 时才允许传 context；详见 recipes/fastwam/train/patch_dataloader_mp_ctx.py\n"
    f"            {MARK} if self.num_workers > 0 else None,\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if not TRAINER.exists():
        print(f"missing {TRAINER}", file=sys.stderr)
        return 2
    src = TRAINER.read_text(encoding="utf-8")

    if args.revert:
        if not BACKUP.exists():
            print(f"no backup at {BACKUP}", file=sys.stderr)
            return 2
        shutil.copyfile(BACKUP, TRAINER)
        print(f"reverted {TRAINER} from {BACKUP}")
        return 0

    applied = MARK in src
    if args.check:
        print(f"dataloader-mp-ctx={'yes' if applied else 'MISSING'}")
        return 0 if applied else 1
    if applied:
        print(f"already patched: {TRAINER}")
        return 0
    if src.count(ANCHOR) != 1:
        raise SystemExit(f"锚点不唯一/缺失：出现 {src.count(ANCHOR)} 次")

    new = src.replace(ANCHOR, NEW, 1)
    compile(new, str(TRAINER), "exec")  # 先验证语法，再落盘
    if not BACKUP.exists():
        shutil.copyfile(TRAINER, BACKUP)
        print(f"backup -> {BACKUP}")
    TRAINER.write_text(new, encoding="utf-8")
    print(f"patched {TRAINER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
