"""τ0-WM 容器自检。卡忙时 SMOKE_DEV=0：只 import，不把张量放到设备上。"""
import os
import sys

print("python", sys.version.split()[0])

import torch

print(
    "torch",
    torch.__version__,
    "file",
    torch.__file__,
    "cuda_api",
    torch.cuda.is_available(),
)

for name in (
    "einops",
    "omegaconf",
    "yaml",
    "diffusers",
    "transformers",
    "accelerate",
    "av",
    "kornia",
    "pandas",
    "pyarrow",
):
    __import__(name)
    print("import", name, "OK")

try:
    import lerobot

    print("lerobot", getattr(lerobot, "__version__", "?"))
except Exception as exc:  # noqa: BLE001
    print("WARN lerobot", type(exc).__name__, exc)

repo = os.environ.get("TAU_REPO", "")
if repo:
    sys.path.insert(0, repo)
    os.chdir(repo)
    import runner.posttrain as posttrain

    print("Trainer", posttrain.Trainer.__name__)

if os.environ.get("SMOKE_DEV", "0") == "1" and torch.cuda.is_available():
    x = torch.randn(8, 16, device="cuda")
    y = x @ x.transpose(0, 1)
    torch.cuda.synchronize()
    print("device_ok", y.device, float(y.detach().float().sum()))

print("TAU_SMOKE_PASS")
