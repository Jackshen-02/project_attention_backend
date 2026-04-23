from __future__ import annotations

import torch

from .common import attention_scale
from .kv_cache import ContiguousKVCache, PagedKVCache


def decode_attention_contiguous(
    q: torch.Tensor,
    cache: ContiguousKVCache,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    if q.dim() != 4 or q.size(-2) != 1:
        raise ValueError("decode_attention_contiguous expects q with shape (batch, heads, 1, head_dim).")

    batch_size, num_heads, _, head_dim = q.shape
    if batch_size != cache.batch_size or num_heads != cache.num_heads or head_dim != cache.head_dim:
        raise ValueError("Query shape does not match contiguous KV cache layout.")

    scale_value = attention_scale(head_dim, scale)
    outputs = torch.zeros_like(q)
    for batch_idx in range(batch_size):
        k_seq, v_seq = cache.sequence(batch_idx)
        scores = torch.matmul(q[batch_idx : batch_idx + 1], k_seq.transpose(-1, -2)) * scale_value
        probs = torch.softmax(scores, dim=-1)
        outputs[batch_idx : batch_idx + 1] = torch.matmul(probs, v_seq)
    return outputs


def decode_attention_paged(
    q: torch.Tensor,
    cache: PagedKVCache,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    if q.dim() != 4 or q.size(-2) != 1:
        raise ValueError("decode_attention_paged expects q with shape (batch, heads, 1, head_dim).")

    batch_size, num_heads, _, head_dim = q.shape
    if batch_size != cache.batch_size or num_heads != cache.num_heads or head_dim != cache.head_dim:
        raise ValueError("Query shape does not match paged KV cache layout.")

    scale_value = attention_scale(head_dim, scale)
    outputs = torch.zeros_like(q)

    for batch_idx in range(batch_size):
        q_seq = q[batch_idx : batch_idx + 1]
        running_max = torch.full(
            (1, num_heads, 1, 1),
            float("-inf"),
            dtype=q.dtype,
            device=q.device,
        )
        running_lse = torch.zeros((1, num_heads, 1, 1), dtype=q.dtype, device=q.device)
        output = torch.zeros_like(q_seq)

        for k_block, v_block in cache.iter_sequence_blocks(batch_idx):
            scores = torch.matmul(q_seq, k_block.transpose(-1, -2)) * scale_value
            block_max = scores.max(dim=-1, keepdim=True).values
            block_probs = torch.exp(scores - block_max)
            block_lse = block_probs.sum(dim=-1, keepdim=True)
            block_value = torch.matmul(block_probs, v_block)

            new_max = torch.maximum(running_max, block_max)
            prev_scale = torch.exp(running_max - new_max)
            block_scale = torch.exp(block_max - new_max)
            new_lse = prev_scale * running_lse + block_scale * block_lse

            numerator = output * (prev_scale * running_lse) + block_value * block_scale
            output = numerator / new_lse.clamp_min(1e-9)
            running_max = new_max
            running_lse = new_lse

        outputs[batch_idx : batch_idx + 1] = output

    return outputs
