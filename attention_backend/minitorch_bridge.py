from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import numpy as np
import torch


@dataclass(frozen=True)
class SharedAttentionWeights:
    x: np.ndarray
    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    w_out: np.ndarray


def build_shared_attention_problem(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    *,
    seed: int,
) -> SharedAttentionWeights:
    n_embd = num_heads * head_dim
    rng = np.random.default_rng(seed)
    proj_std = 1.0 / np.sqrt(n_embd)
    return SharedAttentionWeights(
        x=rng.standard_normal((batch_size, seq_len, n_embd), dtype=np.float32),
        w_q=(rng.standard_normal((n_embd, n_embd), dtype=np.float32) * proj_std).astype(np.float32),
        w_k=(rng.standard_normal((n_embd, n_embd), dtype=np.float32) * proj_std).astype(np.float32),
        w_v=(rng.standard_normal((n_embd, n_embd), dtype=np.float32) * proj_std).astype(np.float32),
        w_out=(rng.standard_normal((n_embd, n_embd), dtype=np.float32) * proj_std).astype(np.float32),
    )


def load_minitorch_modules(device: torch.device):
    try:
        minitorch = import_module("minitorch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MiniTorch dependencies are not installed. Install numpy, numba, torch, "
            "and on GPU runs also install pycuda."
        ) from exc
    if device.type == "cuda":
        try:
            backend_cls = import_module("minitorch.cuda_kernel_ops").CudaKernelOps
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CUDA MiniTorch backend requires pycuda in this environment."
            ) from exc
    else:
        backend_cls = import_module("minitorch.fast_ops").FastOps
    return minitorch, backend_cls


def make_minitorch_backend(device: torch.device):
    minitorch, backend_cls = load_minitorch_modules(device)
    return minitorch, minitorch.TensorBackend(backend_cls)


def build_minitorch_attention_layer(
    shared: SharedAttentionWeights,
    *,
    num_heads: int,
    backend,
    causal: bool,
    attention_backend: str = "naive",
):
    minitorch = import_module("minitorch")
    layer = minitorch.MultiHeadAttention(
        n_embd=shared.x.shape[-1],
        n_head=num_heads,
        causal=causal,
        p_dropout=0.0,
        bias=False,
        backend=backend,
        use_fused_kernel=False,
        attention_backend=attention_backend,
    )
    layer.q_projection.weights.value = minitorch.tensor_from_numpy(shared.w_q, backend=backend)
    layer.k_projection.weights.value = minitorch.tensor_from_numpy(shared.w_k, backend=backend)
    layer.v_projection.weights.value = minitorch.tensor_from_numpy(shared.w_v, backend=backend)
    layer.out_projection.weights.value = minitorch.tensor_from_numpy(shared.w_out, backend=backend)
    return layer


def make_minitorch_input(shared: SharedAttentionWeights, backend):
    minitorch = import_module("minitorch")
    return minitorch.tensor_from_numpy(shared.x, backend=backend)


def make_minitorch_decode_input(
    shared: SharedAttentionWeights,
    positions: list[int] | tuple[int, ...],
    backend,
):
    minitorch = import_module("minitorch")
    batch_size = shared.x.shape[0]
    if len(positions) != batch_size:
        raise ValueError(f"Expected {batch_size} decode positions, got {len(positions)}.")
    x_step = np.stack([shared.x[batch_idx, pos, :] for batch_idx, pos in enumerate(positions)], axis=0)
    return minitorch.tensor_from_numpy(x_step[:, None, :], backend=backend)


def populate_cache_from_prefix(
    layer,
    shared: SharedAttentionWeights,
    prefix_lens: list[int] | tuple[int, ...],
    backend,
    kv_cache,
) -> None:
    max_prefix_len = max(prefix_lens)
    if max_prefix_len == 0:
        return
    minitorch = import_module("minitorch")
    prefix_x = minitorch.tensor_from_numpy(shared.x[:, :max_prefix_len, :], backend=backend)
    _, k_t, v_t = layer._project_to_torch_qkv(prefix_x)
    kv_cache.append(k_t, v_t, valid_lens=prefix_lens)


def torch_project_qkv(
    shared: SharedAttentionWeights,
    *,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_embd = num_heads * head_dim
    x = torch.tensor(shared.x, device=device, dtype=dtype)
    w_q = torch.tensor(shared.w_q, device=device, dtype=dtype)
    w_k = torch.tensor(shared.w_k, device=device, dtype=dtype)
    w_v = torch.tensor(shared.w_v, device=device, dtype=dtype)

    batch_size, seq_len, _ = x.shape
    x_2d = x.view(batch_size * seq_len, n_embd)
    q = torch.matmul(x_2d, w_q).view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    k = torch.matmul(x_2d, w_k).view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    v = torch.matmul(x_2d, w_v).view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    return q, k, v
