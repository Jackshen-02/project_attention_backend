from __future__ import annotations

import torch

from attention_backend.flash import FlashAttentionConfig, flash_attention_tiled
from attention_backend.naive import naive_attention


def _run_case(seq_len: int, causal: bool) -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, seq_len, 8)
    k = torch.randn(2, 3, seq_len, 8)
    v = torch.randn(2, 3, seq_len, 8)

    reference = naive_attention(q, k, v, causal=causal)
    candidate = flash_attention_tiled(
        q,
        k,
        v,
        config=FlashAttentionConfig(block_size=5, causal=causal),
    )

    assert torch.allclose(candidate, reference, atol=1e-5, rtol=1e-5)


def test_flash_attention_matches_naive_noncausal() -> None:
    _run_case(seq_len=13, causal=False)


def test_flash_attention_matches_naive_causal() -> None:
    _run_case(seq_len=17, causal=True)
