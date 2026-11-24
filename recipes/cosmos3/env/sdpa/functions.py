# SPDX-FileCopyrightText: TongBot shim
"""F.scaled_dot_product_attention backend. Layout matches cudnn: heads-last [B,S,H,D]."""

import torch
from torch import Tensor
from torch.nn import functional as F

from cosmos_framework.model.attention.checks import assert_universal_tensor_checks
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.sdpa.checks import sdpa_attention_check


def sdpa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    is_varlen = cumulative_seqlen_Q is not None
    assert_universal_tensor_checks(query, key, value)
    assert sdpa_attention_check(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=is_varlen,
        deterministic=deterministic,
        raise_error=True,
    )
    assert not is_varlen

    scale = scale if scale is not None else query.shape[-1] ** -0.5
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    enable_gqa = q.shape[1] != k.shape[1]
    sdpa_kwargs = dict(is_causal=is_causal, scale=scale)
    if enable_gqa:
        sdpa_kwargs["enable_gqa"] = True

    if return_lse:
        scores = (q.float() @ k.float().transpose(-2, -1)) * scale
        if is_causal:
            nq, nkv = scores.shape[-2], scores.shape[-1]
            causal = torch.ones(nq, nkv, device=scores.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal, float("-inf"))
        lse = scores.logsumexp(dim=-1)
        weights = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        output = weights @ v
        output = output.transpose(1, 2).contiguous()
        lse = lse.transpose(1, 2)
        return output, lse

    output = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
    return output.transpose(1, 2).contiguous()
