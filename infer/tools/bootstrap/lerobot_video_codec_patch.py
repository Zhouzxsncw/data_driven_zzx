"""Monkey-patch get_safe_default_codec when torchcodec is broken (不改 src/lerobot)."""
from __future__ import annotations

import importlib.util
import logging


def _torchcodec_usable() -> bool:
    if importlib.util.find_spec("torchcodec") is None:
        return False
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        return True
    except Exception:
        return False


def apply() -> None:
    from lerobot.datasets import video_utils

    if getattr(video_utils, "_cqtongbot_codec_patch_applied", False):
        return

    def get_safe_default_codec() -> str:
        if _torchcodec_usable():
            return "torchcodec"
        import torchvision_videoreader_shim  # noqa: F401

        logging.warning(
            "'torchcodec' is not available or failed to load, falling back to 'pyav' as a default decoder"
        )
        return "pyav"

    video_utils.get_safe_default_codec = get_safe_default_codec
    video_utils._cqtongbot_codec_patch_applied = True


apply()
