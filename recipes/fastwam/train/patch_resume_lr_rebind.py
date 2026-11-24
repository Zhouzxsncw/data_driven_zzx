#!/usr/bin/env python3
"""给 FastWAM trainer 打「断点续训 LR 冻结」补丁（幂等，带 --check，自动备份）。

背景（实测现象，gpu-master /home/xk/VLA/FastWAM）：
  00036 从 state/step_010000 真续训后，loss 逐步复现历史值（0.3482/0.3042/0.2207），
  但日志里的 lr 一直钉在 5.46e-05，不再按 cosine 衰减（历史同段是 5.45/5.44/5.43e-05）。

根因：
  torch.optim.Optimizer.load_state_dict 走 __setstate__，会用**新的** param_group 字典对象
  替换 optimizer.param_groups；DeepSpeed ZeRO-1 + CPU offload 的内层优化器同理。
  而 LR scheduler 是在 accelerator.prepare / accelerator.load_state **之前**构造的，
  它保存的 `scheduler.optimizer` 引用与 group 字典已不是训练真正读写的那一份，
  于是 scheduler.step() 把 lr 写进孤儿字典，实际生效的 lr 永远停在 ckpt 里存的那个值 ——
  续训后半程（本例 10000→20000 整个 cosine 退火段）等于变成恒定 lr 训练。

修法：
  load_state 之后，把外层及 SequentialLR 内层所有 scheduler 的 `.optimizer` 指回
  self.optimizer（训练循环打印/生效的就是它），并保证活字典里带 initial_lr
  （CosineAnnealingLR.get_values 依赖它；缺失就用 self.learning_rate 补上）。
  这样 restored 的 last_epoch/base_lrs 继续跑原schedule，逐位还原历史 lr 序列。

用法：
  python patch_resume_lr_rebind.py --check     # 只检查是否已打（未打退出码 1）
  python patch_resume_lr_rebind.py             # 打补丁（已打则跳过）
  python patch_resume_lr_rebind.py --revert    # 还原
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().with_suffix("")  # unused; kept for clarity
TRAINER = Path("/home/xk/VLA/FastWAM/src/fastwam/trainer.py")
BACKUP = TRAINER.with_name("trainer.py.orig_pre_lr_rebind")
MARK_BEGIN = "  # --- [fastwam-resume-lr-rebind] begin ---"
MARK_END = "  # --- [fastwam-resume-lr-rebind] end ---"

METHOD_SRC = '''    def _rebind_schedulers_to_live_optimizer(self) -> int:
        """把 resume 前构造的 LR scheduler 重新指回训练真正使用的 optimizer（返回改写条数）。

        `Optimizer.load_state_dict` 会用新的 param_group 字典替换旧字典，导致旧 scheduler
        写进孤儿字典、生效 lr 冻结在 ckpt 里那份数值上。见 recipes/fastwam/train/
        patch_resume_lr_rebind.py 文档字符串。
        """
        target = self.optimizer
        roots = [getattr(self.scheduler, "scheduler", self.scheduler)]
        seen: set[int] = set()
        rebound = 0
        while roots:
            sched = roots.pop()
            if sched is None or id(sched) in seen:
                continue
            seen.add(id(sched))
            if hasattr(sched, "optimizer") and getattr(sched, "optimizer", None) is not target:
                try:
                    sched.optimizer = target
                    rebound += 1
                except Exception:  # noqa: BLE001 - 只读型属性时保持原状即可
                    logger.warning("[resume] 无法改写 scheduler %s 的 optimizer", type(sched).__name__)
            for value in list(getattr(sched, "__dict__", {}).values()):
                for candidate in value if isinstance(value, (list, tuple)) else [value]:
                    if hasattr(candidate, "optimizer") and hasattr(candidate, "get_lr"):
                        roots.append(candidate)

        for index, group in enumerate(target.param_groups):
            if "initial_lr" not in group:
                group["initial_lr"] = float(self.learning_rate)
                logger.warning(
                    "[resume] param_group[%d] 缺 initial_lr，按 peak=%.3e 补齐以驱动 cosine",
                    index,
                    float(self.learning_rate),
                )
        return rebound
'''

HOOK_SRC = f"""{MARK_BEGIN}
        # accelerate+DeepSpeed 的 load_state 会换掉 param_groups 字典，LR schedule 就此冻结。
        _rebound = self._rebind_schedulers_to_live_optimizer()
        logger.info(
            "[resume] re-bound %d scheduler(s) -> live optimizer; lr now %.6e (initial_lr %.6e)",
            _rebound,
            float(self.optimizer.param_groups[0]["lr"]),
            float(self.optimizer.param_groups[0].get("initial_lr", 0.0)),
        )
{MARK_END}
"""

ANCHOR = """    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
"""
METHOD_ANCHOR = "    def load_training_state(self, state_dir: str):"

# ---- 第二个 hunk：resume 内存峰值定位（默认惰性，FASTWAM_RESUME_DIAG=1 才打印）----
MEMDIAG_MARK = "        self._resume_mem_diag(\"after load_state\")  # [fastwam-resume-memdiag]"
MEMDIAG_METHOD = '''    def _resume_mem_diag(self, tag: str) -> None:
        """断点续训内存峰值定位（默认关闭，`FASTWAM_RESUME_DIAG=1` 打开）。

        打印本进程 RssAnon/RssShmem/VmSwap（gc 前后各一次）与优化器状态里 CPU 张量的真实体积，
        用来判断 accelerator.load_state 反序列化的那 72G optim_states 临时张量有没有被回收 ——
        这是 2026-09-14 三次 resume 被 global OOM 打死时唯一缺的那项数据。
        """
        if os.environ.get("FASTWAM_RESUME_DIAG", "0") != "1":
            return

        def _status() -> dict:
            out = {}
            try:
                with open("/proc/self/status", "r", encoding="utf-8") as handle:
                    for line in handle:
                        key = line.split(":", 1)[0]
                        if key in ("RssAnon", "RssShmem", "VmSwap", "VmRSS"):
                            out[key] = int(line.split()[1]) // 1048576
            except OSError:
                pass
            return out

        import gc

        before = _status()
        freed = gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = _status()
        cpu_gib = gpu_gib = 0.0
        n_cpu = n_gpu = 0
        inner = self.optimizer
        for hop in ("optimizer", "_optimizer", "local_optimizer", "_local_secondary_optimizer"):
            inner = getattr(inner, hop, inner)
        state = getattr(inner, "state", None)
        if isinstance(state, dict):
            for per_param in state.values():
                if not isinstance(per_param, dict):
                    continue
                for value in per_param.values():
                    if not torch.is_tensor(value):
                        continue
                    if value.device.type == "cuda":
                        gpu_gib += value.numel() * value.element_size() / 1024**3
                        n_gpu += 1
                    else:
                        cpu_gib += value.numel() * value.element_size() / 1024**3
                        n_cpu += 1
        avail_gib = -1
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable"):
                        avail_gib = int(line.split()[1]) // 1048576
                        break
        except OSError:
            pass
        logger.info(
            "[memdiag] %s RssAnon %s->%sG RssShmem %sG VmSwap %sG gc_freed=%d "
            "| optim cpu_tensors=%d %.1fG gpu_tensors=%d %.1fG | MemAvailable=%sG",
            tag,
            before.get("RssAnon", -1),
            after.get("RssAnon", -1),
            after.get("RssShmem", -1),
            after.get("VmSwap", -1),
            freed,
            n_cpu,
            cpu_gib,
            n_gpu,
            gpu_gib,
            avail_gib,
        )
'''


def patched(src: str) -> bool:
    return MARK_BEGIN in src


def memdiag_applied(src: str) -> bool:
    return MEMDIAG_MARK in src and "def _resume_mem_diag" in src


def would_apply(src: str) -> str:
    if not src.count(METHOD_ANCHOR) == 1:
        raise SystemExit(f"锚点不唯一/缺失：{METHOD_ANCHOR!r} 出现 {src.count(METHOD_ANCHOR)} 次")
    if not patched(src):
        out = src.replace(
            ANCHOR,
            ANCHOR + HOOK_SRC,
            1,
        )
        # 方法体插在 load_training_state 之前（与 HOOK 同一次 replace 会互相干扰，故分两步）
        out = out.replace(METHOD_ANCHOR, METHOD_SRC + METHOD_ANCHOR, 1)
        src = out
    if not memdiag_applied(src):
        src = src.replace(MARK_END + "\n", MARK_END + "\n" + MEMDIAG_MARK + "\n", 1)
        src = src.replace(METHOD_ANCHOR, MEMDIAG_METHOD + METHOD_ANCHOR, 1)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查补丁状态")
    ap.add_argument("--revert", action="store_true", help="还原到备份")
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

    if args.check:
        ok_lr, ok_diag = patched(src), memdiag_applied(src)
        print(f"lr-rebind={'yes' if ok_lr else 'MISSING'} memdiag={'yes' if ok_diag else 'MISSING'}")
        return 0 if (ok_lr and ok_diag) else 1

    if patched(src) and memdiag_applied(src):
        print(f"already patched: {TRAINER}")
        return 0

    new = would_apply(src)
    compile(new, str(TRAINER), "exec")  # 先验证语法，再落盘
    if not BACKUP.exists():
        shutil.copyfile(TRAINER, BACKUP)
        print(f"backup -> {BACKUP}")
    TRAINER.write_text(new, encoding="utf-8")
    print(f"patched {TRAINER} (+{len(HOOK_SRC.splitlines()) + len(METHOD_SRC.splitlines())} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
