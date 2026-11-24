"""transformers 4.53.x 兼容 GR00T Eagle fast image processor（不改 src/lerobot）。"""
from __future__ import annotations


def apply_groot_eagle_processor_compat() -> None:
    try:
        from transformers.image_processing_utils_fast import BaseImageProcessorFast
    except ImportError:
        return
    if hasattr(BaseImageProcessorFast, "_prepare_image_like_inputs"):
        return
    if not hasattr(BaseImageProcessorFast, "_prepare_input_images"):
        return

    def _prepare_image_like_inputs(
        self,
        images,
        do_convert_rgb=None,
        input_data_format=None,
        device=None,
        expected_ndims=3,
        **kwargs,
    ):
        return self._prepare_input_images(
            images=images,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )

    BaseImageProcessorFast._prepare_image_like_inputs = _prepare_image_like_inputs
