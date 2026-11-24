#!/usr/bin/env python3
"""给 X-VLA 的上游 train.py 打上游未支持的 --resume 断点续训补丁。

背景：上游 train.py 把 global_step 硬编码为 0，且 checkpoint 只保存模型权重
（save_pretrained + state.json{global_step}），不保存 optimizer / RNG 状态。
因此直接 `--models ckpt-5000` 会把 freeze_steps/warmup/cosine 整套 LR schedule
从头再走一遍，等于不是续训。

本补丁加 `--resume <ckpt_dir>`：
  1) 权重与 processor/tokenizer 从 ckpt 目录加载（顺带保证 action_mode /
     real_action_dim 等取自 ckpt 自身 config，不会被 --models 换掉）；
  2) 从 ckpt/state.json 恢复 global_step，使 LR schedule 与 ckpt 命名连续。
AdamW 矩量仍从 0 重启（上游不存 optimizer，且补存会把磁盘写爆：约 14G/ckpt）。
幂等：重复执行不会二次打补丁。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TRAIN_PY = Path("/home/xk/VLA/X-VLA/train.py")

# (待替换原文, 替换后文本)
REPLACEMENTS = [
    # 1) 新增 --resume 参数
    (
        '''    parser.add_argument("--output_dir", type=str, default="runnings", help="Directory to save checkpoints")
''',
        '''    parser.add_argument("--output_dir", type=str, default="runnings", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 ckpt-* 目录续训：加载该 ckpt 权重，并把 global_step 恢复到其 state.json 记录的步数")
''',
    ),
    # 2) 权重加载改走 ckpt，并恢复 global_step
    (
        '''    # Load model & processor
    model = XVLA.from_pretrained(args.models)
    processor = XVLAProcessor.from_pretrained(args.models)
''',
        '''    # Load model & processor（续训时用 ckpt 自带的 config/tokenizer，避免 action_mode 等被 --models 换掉）
    load_path = args.resume if args.resume else args.models
    model = XVLA.from_pretrained(load_path)
    processor = XVLAProcessor.from_pretrained(load_path)

    # 恢复 global_step：让 freeze/warmup/cosine 按原进度继续，而不是从头再走一遍
    start_step = 0
    if args.resume:
        state_file = os.path.join(args.resume, "state.json")
        if not os.path.isfile(state_file):
            raise FileNotFoundError(f"resume 目录缺少 state.json: {state_file}")
        with open(state_file) as f:
            start_step = int(json.load(f).get("global_step", 0))
        if start_step >= args.iters:
            raise ValueError(f"resume step {start_step} >= iters {args.iters}，没有剩余训练步数")
        logger.warning(
            f"⏩ Resume from {args.resume} @ step {start_step}; "
            f"ckpt 只存权重，AdamW 一/二阶矩从 0 重启（LR schedule 不受影响）"
        )
''',
    ),
    # 3) 训练循环起点改为 start_step
    (
        '''    global_step, t0 = 0, time.time()
    logger.info(f"🚀 Start training for {args.iters} iterations | world_size={accelerator.num_processes}")
''',
        '''    global_step, t0 = start_step, time.time()
    logger.info(f"🚀 Start training from step {global_step} to {args.iters} | world_size={accelerator.num_processes}")
''',
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=TRAIN_PY)
    ap.add_argument("--check", action="store_true", help="只检查是否需要/能否打补丁，不写文件")
    args = ap.parse_args()

    if not args.file.is_file():
        print(f"[FAIL] 找不到 {args.file}", file=sys.stderr)
        return 1

    src = original = args.file.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        n = src.count(old)
        if n == 0:
            if new in src:
                continue  # 该处已打过补丁
            print(f"[FAIL] 未匹配到待替换片段（train.py 可能已被改动）：\n{old[:80]}...", file=sys.stderr)
            return 1
        if n > 1:
            print(f"[FAIL] 片段重复 {n} 次，无法唯一定位：{old[:80]}...", file=sys.stderr)
            return 1
        src = src.replace(old, new, 1)

    if src == original:
        print(f"[SKIP] {args.file} 已含 --resume 补丁，无需重复")
        return 0
    if args.check:
        print(f"[NEED] {args.file} 需要打补丁（3 处均可唯一定位）")
        return 0

    compile(src, str(args.file), "exec")  # 写盘前先语法自检
    bak = args.file.with_suffix(".py.orig_pre_resume")
    if not bak.exists():
        bak.write_text(original, encoding="utf-8")
    args.file.write_text(src, encoding="utf-8")
    print(f"[OK] 已打补丁：{args.file}（原文备份 {bak}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
