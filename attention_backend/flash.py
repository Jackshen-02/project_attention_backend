from __future__ import annotations

from dataclasses import dataclass

import torch

from .common import AttentionShape, attention_scale, causal_mask, tensor_nbytes


@dataclass(frozen=True)
class FlashAttentionConfig:
    block_size: int = 128
    causal: bool = True
    scale: float | None = None


def flash_attention_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    config: FlashAttentionConfig | None = None,
) -> torch.Tensor:
    cfg = config or FlashAttentionConfig()
    batch_size, num_heads, query_len, head_dim = q.shape
    key_len = k.size(-2)
    scale = attention_scale(head_dim, cfg.scale)

    output = torch.zeros_like(q)
    running_max = torch.full(
        (batch_size, num_heads, query_len, 1),
        float("-inf"),
        device=q.device,
        dtype=q.dtype,
    )
    running_lse = torch.zeros(
        (batch_size, num_heads, query_len, 1),
        device=q.device,
        dtype=q.dtype,
    )

    for key_start in range(0, key_len, cfg.block_size):
        key_end = min(key_start + cfg.block_size, key_len)
        k_block = k[:, :, key_start:key_end, :]
        v_block = v[:, :, key_start:key_end, :]

        block_scores = torch.matmul(q, k_block.transpose(-1, -2)) * scale
        if cfg.causal:
            block_scores = block_scores.masked_fill(
                causal_mask(query_len, key_start, key_end, device=q.device),
                float("-inf"),
            )

        block_max = block_scores.max(dim=-1, keepdim=True).values
        valid_rows = torch.isfinite(block_max)
        safe_block_max = torch.where(valid_rows, block_max, torch.zeros_like(block_max))
        block_probs = torch.where(
            valid_rows,
            torch.exp(block_scores - safe_block_max),
            torch.zeros_like(block_scores),
        )
        block_lse = block_probs.sum(dim=-1, keepdim=True)
        block_value = torch.matmul(block_probs, v_block)

        new_max = torch.where(valid_rows, torch.maximum(running_max, safe_block_max), running_max)
        prev_scale = torch.exp(running_max - new_max)
        block_scale = torch.where(
            valid_rows,
            torch.exp(safe_block_max - new_max),
            torch.zeros_like(new_max),
        )
        new_lse = prev_scale * running_lse + block_scale * block_lse

        numerator = output * (prev_scale * running_lse) + block_value * block_scale
        output = torch.where(new_lse > 0.0, numerator / new_lse.clamp_min(1e-9), output)
        running_max = new_max
        running_lse = new_lse

    return output


def tiled_peak_intermediate_bytes(
    shape: AttentionShape,
    dtype: torch.dtype,
    block_size: int,
) -> int:
    tile_len = min(shape.key_len, block_size)
    score_shape = (shape.batch_size, shape.num_heads, shape.query_len, tile_len)
    stats_shape = (shape.batch_size, shape.num_heads, shape.query_len, 1)
    output_shape = shape.output_shape
    return (
        2 * tensor_nbytes(score_shape, dtype)
        + 2 * tensor_nbytes(stats_shape, dtype)
        + tensor_nbytes(output_shape, dtype)
    )
