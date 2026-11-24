#!/usr/bin/env python3
"""Patch sharded_mixture_dataset._get_default_pg_tensor_device for Ascend HCCL."""
from pathlib import Path

p = Path("/home/gr00t/src/Isaac-GR00T/gr00t/data/dataset/sharded_mixture_dataset.py")
text = p.read_text()
old = '''def _get_default_pg_tensor_device() -> torch.device:
    """Return a tensor device supported by the default process-group backend."""
    try:
        backend = dist.get_backend()
    except (AssertionError, RuntimeError, ValueError):
        return torch.device("cpu")

    if str(backend).lower() == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ShardedMixtureDataset seed check requires CUDA tensors when the "
                "distributed process group uses NCCL, but CUDA is not available."
            )
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
'''
new = '''def _get_default_pg_tensor_device() -> torch.device:
    """Return a tensor device supported by the default process-group backend."""
    try:
        backend = str(dist.get_backend()).lower()
    except (AssertionError, RuntimeError, ValueError):
        return torch.device("cpu")

    # Ascend transfer_to_npu remaps nccl -> hccl; both need device tensors.
    if backend in ("nccl", "hccl") or backend.startswith("hccl"):
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu", torch.npu.current_device())
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ShardedMixtureDataset seed check requires CUDA/NPU tensors when the "
                "distributed process group uses NCCL/HCCL, but neither is available."
            )
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
'''
if "backend.startswith(\"hccl\")" in text or "backend.startswith('hccl')" in text:
    print("ALREADY_PATCHED")
    raise SystemExit(0)
if old not in text:
    raise SystemExit("PATCH_MISS")
p.write_text(text.replace(old, new))
# drop pyc
for c in p.parent.rglob("__pycache__"):
    import shutil
    shutil.rmtree(c, ignore_errors=True)
print("PATCHED", p)
