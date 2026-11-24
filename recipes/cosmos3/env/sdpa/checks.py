# SPDX-FileCopyrightText: TongBot shim
"""Eligibility checks for the SDPA fallback backend."""

from functools import partial

import torch

from cosmos_framework.model.attention.checks import attention_param_checks, attention_tensor_checks
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.utils import log_or_raise_error


def sdpa_attention_check(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType | None,
    is_varlen: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    target_fn = partial(log_or_raise_error, raise_error=raise_error)
    if is_varlen:
        target_fn("SDPA backend does not support packed varlen attention.", exception=RuntimeError)
        return False
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        target_fn(f"SDPA backend does not support dtype={dtype}.", exception=RuntimeError)
        return False
    if not attention_tensor_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        raise_error=raise_error,
        backend_name="SDPA",
    ):
        return False
    attention_param_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        is_causal=is_causal,
        causal_type=causal_type,
    )
    return True
