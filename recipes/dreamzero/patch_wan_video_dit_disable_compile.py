#!/usr/bin/env python3
"""给 h100-103 上 wan_video_dit.py 的 DISABLE_TORCH_COMPILE 加独立环境变量开关。

在远端执行：python3 patch_wan_video_dit_disable_compile.py

原逻辑：DISABLE_TORCH_COMPILE 只由 ENABLE_TENSORRT 控制（开 TRT 才关 compile，
但会连带把 flash attention 降级成 SDPA）。
新逻辑：DISABLE_TORCH_COMPILE 可单独用环境变量打开，不动 flash attention。
默认（不设环境变量）行为与原来完全一致。

背景：PyTorch 2.9 下 inductor CUDA graphs + FSDP2 梯度检查点显存异常膨胀——
6 卡无 offload 训练 79G/卡贴满仍 OOM，no-reentrant gc 直接触发
cudagraph_trees "Expected curr_block->next == nullptr" 崩溃（2026-09-11 实测）。
"""

P = "/data1/dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit.py"

OLD = '''ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"
if ENABLE_TENSORRT:
    # disable torch compile and transformer engine and flash attention for onnx/tensorrt export
    FLASH_ATTN_COMPATIBILITY_MODE = True
    DISABLE_TORCH_COMPILE = True
else:
    DISABLE_TORCH_COMPILE = False
    FLASH_ATTN_COMPATIBILITY_MODE = False'''

NEW = '''ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"
# DISABLE_TORCH_COMPILE: independent env override (default off = keep torch.compile).
# PyTorch 2.9 inductor CUDA graphs + FSDP2 grad-checkpointing inflates memory
# (6-GPU run: 79G/card OOM + cudagraph_trees errors). Set DISABLE_TORCH_COMPILE=true
# to disable compilation WITHOUT touching flash attention (unlike ENABLE_TENSORRT).
DISABLE_TORCH_COMPILE = (
    os.getenv("DISABLE_TORCH_COMPILE", "False").lower() == "true" or ENABLE_TENSORRT
)
FLASH_ATTN_COMPATIBILITY_MODE = ENABLE_TENSORRT
print(
    f"[wan_video_dit] ENABLE_TENSORRT={ENABLE_TENSORRT} "
    f"DISABLE_TORCH_COMPILE={DISABLE_TORCH_COMPILE} "
    f"FLASH_ATTN_COMPATIBILITY_MODE={FLASH_ATTN_COMPATIBILITY_MODE}",
    flush=True,
)'''


def main() -> None:
    with open(P, encoding="utf-8") as f:
        src = f.read()

    n = src.count(OLD)
    if n == 0:
        if "independent env override" in src:
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
