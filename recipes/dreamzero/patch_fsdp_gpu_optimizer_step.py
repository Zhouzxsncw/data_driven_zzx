#!/usr/bin/env python3
"""给 h100-103 上 RLinf fsdp_model_manager.py 的 optimizer_step 加 GPU onload 包装。

在远端执行：python3 patch_fsdp_gpu_optimizer_step.py

问题：FSDP2 CPUOffloadPolicy（cpu_offload=True）下参数/梯度/优化器状态都在 CPU，
AdamW 走 _single_tensor_adam 单核慢路径——py-spy 实测 14B 分片模型 ~20s/step
（worker 单核 98%，GPU 85% 时间空闲）。

修复：cpu_offload=True 时，optimizer step 前把参数+梯度+优化器状态 onload 到
GPU（GPU foreach AdamW 毫秒级），step 后 offload 回 CPU。PCIe 搬运 ~30G/step
≈ 2s，net 提速 ~18s/step。默认路径（cpu_offload=False）完全不变。
"""

P = "/data1/RLinf/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py"

OLD = '''        self.optimizer_steps += 1
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = self._strategy.clip_grad_norm_(
            model=self.model,
        )

        if not torch.isfinite(torch.as_tensor(grad_norm)):
            self._logger.warning(
                f"[FSDP] Non-finite grad norm {grad_norm} detected. Skipping optimizer step."
            )
        else:
            self.grad_scaler.step(optimizer=self.optimizer)

        self.grad_scaler.update()'''

NEW = '''        self.optimizer_steps += 1

        # With FSDP2 CPUOffloadPolicy, params/grads/optimizer states live on CPU,
        # where AdamW falls back to the single-tensor slow path (~20s/step for a
        # 14B sharded model, one CPU core at 100% while GPUs idle). Temporarily
        # onload everything to GPU, run the step there (milliseconds), then
        # offload back. ~30G/step over PCIe costs ~2s -- far cheaper than CPU AdamW.
        cpu_offload = self._cfg.fsdp_config.get("cpu_offload", False)
        if cpu_offload:
            self.load_param_and_grad(torch.cuda.current_device(), load_grad=True)
            self.load_optimizer(torch.cuda.current_device())

        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = self._strategy.clip_grad_norm_(
            model=self.model,
        )

        if not torch.isfinite(torch.as_tensor(grad_norm)):
            self._logger.warning(
                f"[FSDP] Non-finite grad norm {grad_norm} detected. Skipping optimizer step."
            )
        else:
            self.grad_scaler.step(optimizer=self.optimizer)

        self.grad_scaler.update()

        if cpu_offload:
            self.offload_param_and_grad(offload_grad=False)
            self.offload_optimizer()'''


def main() -> None:
    with open(P, encoding="utf-8") as f:
        src = f.read()

    n = src.count(OLD)
    if n == 0:
        if "Temporarily" in src and "cpu_offload = self._cfg.fsdp_config.get" in src:
            print("SKIP: patch already applied")
            return
        raise SystemExit("ERROR: target block not found (upstream changed?)")
    if n != 1:
        raise SystemExit(f"ERROR: expected exactly 1 occurrence, found {n}")

    with open(P, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    print("patched OK")

    import ast

    ast.parse(open(P, encoding="utf-8").read())
    print("syntax OK")


if __name__ == "__main__":
    main()
