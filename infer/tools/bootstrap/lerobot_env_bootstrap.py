"""训练/验证前导入：VideoReader shim + 视频 codec 回退（不改 src/lerobot）。"""
from __future__ import annotations

import os

# DataLoader num_workers>0 会 fork；主进程若已用过 tokenizers 会刷屏警告。须在 import 前设置。
# 强制 false：setdefault 无法覆盖 shell 里残留的 TOKENIZERS_PARALLELISM=true。
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bootstrap() -> None:
    from tools.bootstrap import groot_eagle_processor_compat  # noqa: F401
    from tools.bootstrap import torchvision_videoreader_shim  # noqa: F401

    groot_eagle_processor_compat.apply_groot_eagle_processor_compat()

    # `lerobot_video_codec_patch` imports lerobot.datasets.video_utils eagerly.
    # On the PPU runtime that import before Draccus parses PI0.5's config can abort
    # in native code (`std::bad_alloc`). Real/local datasets explicitly use PyAV,
    # so their VideoReader shim above is sufficient and this patch is unnecessary.
    if os.environ.get("SIM_LOCAL_DATASET") != "1":
        from tools.bootstrap import lerobot_video_codec_patch  # noqa: F401
