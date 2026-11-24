# 自动加载 torch_npu 的 cuda→npu 透明映射。
# 华为 pi05 镜像是 torch 2.x+cpu：不 import 这一行，train_pytorch.setup_ddp
# 会因 cuda.is_available()=False 把 8 个 rank 全放到 CPU，npu-smi 看起来像堆在一张卡上。
# 放到 /home/pi05/openpi_src/src/（START 的 PYTHONPATH 已含 src），解释器启动即生效。
try:
    import torch_npu.contrib.transfer_to_npu  # noqa: F401
except Exception:
    pass
