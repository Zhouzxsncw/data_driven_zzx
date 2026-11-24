#!/usr/bin/env python3
"""V11 补丁：AdamW moments 常驻 CPU、optimizer step 时往返 GPU。

在远端执行：python3 patch_v11_optim_offload.py

背景（2026-09-11，6 卡 DreamZero-14B SFT 排障结论）：
- 6 卡无 offload：backward 峰值 74.83G allocated + 非 PyTorch ~4G = 79.06G，恒定
  OOM（V2/V3/V4/V7/V7b 五组合验证）。峰值里含常驻 AdamW moments（bf16 ~9.3G/rank）。
- CPUOffloadPolicy（cpu_offload=True）：能跑但 AdamW 走 CPU _single_tensor_adam
  单核慢路径 ~20s/step（py-spy 实锤），总 26.75s/step。
- V10（cpu_offload + step 前 model.to(device) onload 参数）：FSDP2 storage-swap
  RuntimeError —— worker 层参数 onload 与 CPUOffloadPolicy 不兼容。

V11 方案：参数/梯度 GPU 常驻（无 CPUOffloadPolicy），只 offload AdamW moments：
  init/首次 run_training 时 offload_optimizer()（含 resume 恢复的状态）
  → optimizer_step 前 load_optimizer(cuda)（GPU foreach AdamW，毫秒级）
  → step 后 offload_optimizer() 回 CPU
净效果：backward 峰值 ~65.5G（<79.18G，余 13G）；step 总耗时估 5-6s
（vs CPUOffloadPolicy 的 26.75s）。开关：actor.enable_offload=True。
"""

import ast

MANAGER = "/data1/RLinf/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py"
WORKER = "/data1/RLinf/rlinf/workers/sft/fsdp_sft_worker.py"

# ---- Patch 1: fsdp_model_manager.py optimizer_step（替换 V10 的 cpu_offload 版本）----
M_OLD = """        self.optimizer_steps += 1

        # With FSDP2 CPUOffloadPolicy, params/grads/optimizer states live on CPU,
        # where AdamW falls back to the single-tensor slow path (~20s/step for a
        # 14B sharded model, one CPU core at 100% while GPUs idle). Temporarily
        # onload everything to GPU, run the step there (milliseconds), then
        # offload back. ~30G/step over PCIe costs ~2s -- far cheaper than CPU AdamW.
        cpu_offload = self._cfg.fsdp_config.get("cpu_offload", False)
        if cpu_offload:
            self.load_param_and_grad(torch.cuda.current_device(), load_grad=True)
            self.load_optimizer(torch.cuda.current_device())
"""

M_NEW = """        self.optimizer_steps += 1

        # Optimizer-state offload (actor.enable_offload=True): AdamW moments live
        # on CPU between steps; onload to GPU just for the step (GPU foreach AdamW
        # = milliseconds) and offload back afterwards. Frees ~10G/rank of steady
        # VRAM (lets 6xH100 fit a 14B full_shard run) and avoids the CPU
        # single-tensor AdamW slow path (~20s/step). Params/grads stay on GPU:
        # model.to() onload is incompatible with FSDP2 (storage-swap error) and
        # unneeded when CPUOffloadPolicy is off.
        optim_offload = self._cfg.get("enable_offload", False)
        if optim_offload:
            self.load_optimizer(torch.cuda.current_device())
"""

M_OLD2 = """        self.grad_scaler.update()

        if cpu_offload:
            self.offload_param_and_grad(offload_grad=False)
            self.offload_optimizer()"""

M_NEW2 = """        self.grad_scaler.update()

        if optim_offload:
            self.offload_optimizer()"""

# ---- Patch 2: fsdp_sft_worker.py init_worker + run_training 兜底 ----
W_OLD = """    def init_worker(self):
        self.setup_model_and_optimizer()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()"""

W_NEW = """    def init_worker(self):
        self.setup_model_and_optimizer()

        if self.cfg.actor.get("enable_offload", False):
            # Only offload optimizer states. Offloading params/grads via
            # model.to() breaks under FSDP2 (storage-swap RuntimeError, verified
            # 2026-09-11); params offload should use fsdp_config.cpu_offload
            # (CPUOffloadPolicy) instead, at the cost of slow CPU AdamW.
            self.offload_optimizer()"""

W_OLD2 = """    def run_training(self):
        with self.worker_timer():
            self.model.train()
"""

W_NEW2 = """    def run_training(self):
        with self.worker_timer():
            self.model.train()

            # enable_offload: ensure AdamW moments are off the GPU before the
            # first backward. Also covers optimizer states restored by resume,
            # which may land on GPU after init_worker already ran.
            if self.cfg.actor.get("enable_offload", False) and not getattr(
                self, "_optim_offload_done", False
            ):
                self.offload_optimizer()
                self._optim_offload_done = True
"""


def apply(path: str, old: str, new: str, tag: str) -> None:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    n = src.count(old)
    if n == 1:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src.replace(old, new))
        ast.parse(open(path, encoding="utf-8").read())
        print(f"{tag}: patched OK + syntax OK")
    elif n == 0:
        if new.strip().splitlines()[-1] in src or new.splitlines()[0] in src:
            print(f"{tag}: SKIP (already applied)")
        else:
            raise SystemExit(f"{tag}: ERROR target block not found")
    else:
        raise SystemExit(f"{tag}: ERROR {n} occurrences, expected 1")


def main() -> None:
    apply(MANAGER, M_OLD, M_NEW, "manager/optimizer_step-head")
    apply(MANAGER, M_OLD2, M_NEW2, "manager/optimizer_step-tail")
    apply(WORKER, W_OLD, W_NEW, "worker/init_worker")
    apply(WORKER, W_OLD2, W_NEW2, "worker/run_training")


if __name__ == "__main__":
    main()
