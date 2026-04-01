from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
try:
    from numba import cuda as numba_cuda
except Exception:  # pragma: no cover
    numba_cuda = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_backend.minitorch_bridge import build_minitorch_attention_layer, build_shared_attention_problem, make_minitorch_backend


pytestmark = pytest.mark.skipif(
    numba_cuda is None or not numba_cuda.is_available() or not torch.cuda.is_available(),
    reason="MiniTorch attention alignment is validated on CUDA backends only",
)


def test_minitorch_naive_attention_matches_torch_reference() -> None:
    minitorch = importlib.import_module("minitorch")
    device = torch.device("cuda")
    shared = build_shared_attention_problem(
        batch_size=2,
        seq_len=7,
        num_heads=2,
        head_dim=4,
        seed=0,
    )
    _, backend = make_minitorch_backend(device)
    layer = build_minitorch_attention_layer(shared, num_heads=2, backend=backend, causal=True)
    x = minitorch.tensor_from_numpy(shared.x, backend=backend)

    result = layer(x).to_numpy()

    x_t = torch.tensor(shared.x, dtype=torch.float32, device=device)
    w_q = torch.tensor(shared.w_q, dtype=torch.float32, device=device)
    w_k = torch.tensor(shared.w_k, dtype=torch.float32, device=device)
    w_v = torch.tensor(shared.w_v, dtype=torch.float32, device=device)
    w_out = torch.tensor(shared.w_out, dtype=torch.float32, device=device)

    q = (x_t.view(-1, 8) @ w_q).view(2, 7, 2, 4).permute(0, 2, 1, 3)
    k = (x_t.view(-1, 8) @ w_k).view(2, 7, 2, 4).permute(0, 2, 1, 3)
    v = (x_t.view(-1, 8) @ w_v).view(2, 7, 2, 4).permute(0, 2, 1, 3)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (4 ** 0.5)
    mask = torch.triu(torch.ones(7, 7, dtype=torch.bool, device=device), diagonal=1)
    scores = scores.masked_fill(mask.view(1, 1, 7, 7), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(2, 7, 8)
    reference = torch.matmul(out.view(-1, 8), w_out).view(2, 7, 8).detach().cpu().numpy()

    np.testing.assert_allclose(result, reference, atol=1e-5, rtol=1e-5)


def test_minitorch_flash_backend_matches_minitorch_naive() -> None:
    minitorch = importlib.import_module("minitorch")
    device = torch.device("cuda")
    shared = build_shared_attention_problem(
        batch_size=2,
        seq_len=9,
        num_heads=2,
        head_dim=4,
        seed=1,
    )
    _, backend = make_minitorch_backend(device)
    x = minitorch.tensor_from_numpy(shared.x, backend=backend)

    naive_layer = build_minitorch_attention_layer(
        shared,
        num_heads=2,
        backend=backend,
        causal=True,
        attention_backend="naive",
    )
    flash_layer = build_minitorch_attention_layer(
        shared,
        num_heads=2,
        backend=backend,
        causal=True,
        attention_backend="flash_tiled",
    )

    naive_out = naive_layer(x).to_numpy()
    flash_out = flash_layer(x).to_numpy()

    np.testing.assert_allclose(flash_out, naive_out, atol=1e-5, rtol=1e-5)
