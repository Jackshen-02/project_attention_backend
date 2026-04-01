from __future__ import annotations

import torch

from .common import AttentionShape, attention_scale, tensor_nbytes


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    head_dim = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * attention_scale(head_dim, scale)

    if causal:
        query_len = q.size(-2)
        key_len = k.size(-2)
        mask = torch.triu(
            torch.ones(query_len, key_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask.view(1, 1, query_len, key_len), float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def naive_peak_intermediate_bytes(shape: AttentionShape, dtype: torch.dtype) -> int:
    score_shape = (shape.batch_size, shape.num_heads, shape.query_len, shape.key_len)
    return 2 * tensor_nbytes(score_shape, dtype)
