from __future__ import annotations

import numpy as np
import pytest
import torch

from attention_backend.kv_cache import ContiguousKVCache, PagedKVCache
from attention_backend.minitorch_bridge import (
    build_minitorch_attention_layer,
    build_shared_attention_problem,
    make_minitorch_backend,
    make_minitorch_decode_input,
    populate_cache_from_prefix,
)


def test_contiguous_decode_matches_full_forward_last_token() -> None:
    pytest.importorskip("numba")
    minitorch, backend = make_minitorch_backend(torch.device("cpu"))
    shared = build_shared_attention_problem(
        batch_size=1,
        seq_len=6,
        num_heads=2,
        head_dim=4,
        seed=7,
    )
    layer = build_minitorch_attention_layer(shared, num_heads=2, backend=backend, causal=True)

    prefix_len = 4
    cache = ContiguousKVCache.allocate(
        batch_size=1,
        num_heads=2,
        head_dim=4,
        max_cache_len=6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    populate_cache_from_prefix(layer, shared, [prefix_len], backend, cache)

    x_step = make_minitorch_decode_input(shared, [prefix_len], backend)
    decode_out = layer.decode_step(x_step, cache, cache_backend="contiguous").to_numpy()

    full_x = minitorch.tensor_from_numpy(shared.x[:, : prefix_len + 1, :], backend=backend)
    forward_out = layer(full_x).to_numpy()[:, -1:, :]
    np.testing.assert_allclose(decode_out, forward_out, atol=1e-5, rtol=1e-5)


def test_paged_decode_matches_contiguous_decode() -> None:
    pytest.importorskip("numba")
    _, backend = make_minitorch_backend(torch.device("cpu"))
    initial_lens = [3, 5]
    decode_steps = 2
    shared = build_shared_attention_problem(
        batch_size=2,
        seq_len=max(initial_lens) + decode_steps,
        num_heads=2,
        head_dim=4,
        seed=11,
    )
    layer = build_minitorch_attention_layer(shared, num_heads=2, backend=backend, causal=True)

    contiguous_cache = ContiguousKVCache.allocate(
        batch_size=2,
        num_heads=2,
        head_dim=4,
        max_cache_len=max(initial_lens) + decode_steps,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    paged_cache = PagedKVCache.allocate(
        batch_size=2,
        num_heads=2,
        head_dim=4,
        page_size=2,
        max_pages=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    populate_cache_from_prefix(layer, shared, initial_lens, backend, contiguous_cache)
    populate_cache_from_prefix(layer, shared, initial_lens, backend, paged_cache)

    for step in range(decode_steps):
        positions = [initial_lens[batch_idx] + step for batch_idx in range(len(initial_lens))]
        x_step = make_minitorch_decode_input(shared, positions, backend)
        contiguous_out = layer.decode_step(x_step, contiguous_cache, cache_backend="contiguous").to_numpy()
        paged_out = layer.decode_step(x_step, paged_cache, cache_backend="paged").to_numpy()
        np.testing.assert_allclose(paged_out, contiguous_out, atol=1e-5, rtol=1e-5)


def test_paged_cache_uses_fewer_bytes_for_variable_lengths() -> None:
    contiguous = ContiguousKVCache.allocate(
        batch_size=2,
        num_heads=2,
        head_dim=4,
        max_cache_len=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    paged = PagedKVCache.allocate(
        batch_size=2,
        num_heads=2,
        head_dim=4,
        page_size=2,
        max_pages=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    k = torch.randn(2, 2, 5, 4)
    v = torch.randn(2, 2, 5, 4)
    contiguous.append(k, v, valid_lens=[1, 5])
    paged.append(k, v, valid_lens=[1, 5])

    assert paged.active_bytes() == contiguous.active_bytes()
    assert paged.allocated_bytes() < contiguous.allocated_bytes()
    assert 0.0 < paged.utilization() <= 1.0
