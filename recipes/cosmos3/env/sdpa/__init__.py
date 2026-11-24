# SPDX-FileCopyrightText: TongBot shim
"""PyTorch SDPA attention backend for PPU/NPU (no flash-attn / cuDNN)."""

from cosmos_framework.model.attention.sdpa.checks import sdpa_attention_check
from cosmos_framework.model.attention.sdpa.functions import sdpa_attention

__all__ = ["sdpa_attention", "sdpa_attention_check"]
