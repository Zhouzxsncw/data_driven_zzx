# 自动加载 torch_npu 的 cuda->npu 透明映射。
try:
    import torch_npu.contrib.transfer_to_npu  # noqa: F401
except Exception:
    pass
