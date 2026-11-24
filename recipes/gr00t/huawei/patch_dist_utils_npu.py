#!/usr/bin/env python3
"""Patch gr00t dist_utils._collective_device for Ascend HCCL."""
from pathlib import Path

p = Path("/home/gr00t/src/Isaac-GR00T/gr00t/utils/dist_utils.py")
text = p.read_text()
marker = "def _collective_device() -> torch.device:"
if "torch.npu.is_available()" in text and "_collective_device" in text:
    print("ALREADY_PATCHED")
    raise SystemExit(0)

start = text.find(marker)
if start < 0:
    raise SystemExit("NO_FUNC")
# find next def after this one
rest = text[start + len(marker) :]
next_def = rest.find("\n@contextmanager")
if next_def < 0:
    next_def = rest.find("\ndef ")
end = start + len(marker) + next_def

new_fn = '''def _collective_device() -> torch.device:
    """Pick the device used for status-broadcast collectives.

    NCCL/HCCL require device tensors (not CPU). Ascend transfer_to_npu remaps
    cuda->npu and nccl->hccl; prefer explicit npu: when available.
    """
    backend = str(torch.distributed.get_backend())
    if backend in ("nccl", "hccl") or backend.startswith("hccl"):
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device(f"npu:{torch.npu.current_device()}")
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


'''
p.write_text(text[:start] + new_fn + text[end:])
print("PATCHED", p)
