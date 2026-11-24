import os
import torch
import torch_npu.contrib.transfer_to_npu  # noqa: F401
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
backend = str(dist.get_backend())
print(f"rank={dist.get_rank()} backend={backend!r} current={torch.cuda.current_device()}", flush=True)
if backend in ("nccl", "hccl") or backend.startswith("hccl"):
    if hasattr(torch, "npu") and torch.npu.is_available():
        d = torch.device(f"npu:{torch.npu.current_device()}")
    else:
        d = torch.device(f"cuda:{torch.cuda.current_device()}")
else:
    d = torch.device("cpu")
t = torch.zeros(1, dtype=torch.int, device=d)
print(f"rank={dist.get_rank()} tensor={t.device} type={t.device.type}", flush=True)
dist.all_reduce(t, op=dist.ReduceOp.MAX)
print(f"rank={dist.get_rank()} all_reduce_OK val={t.item()}", flush=True)
dist.destroy_process_group()
