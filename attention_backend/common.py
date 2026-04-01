from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


def attention_scale(head_dim: int, scale: float | None = None) -> float:
    if scale is not None:
        return scale
    return 1.0 / math.sqrt(head_dim)


def causal_mask(
    query_len: int,
    key_start: int,
    key_end: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    q_positions = torch.arange(query_len, device=device).view(1, 1, query_len, 1)
    k_positions = torch.arange(key_start, key_end, device=device).view(1, 1, 1, key_end - key_start)
    return k_positions > q_positions


@dataclass(frozen=True)
class ErrorStats:
    max_abs: float | None
    max_rel: float | None
    candidate_nonfinite_count: int
    reference_nonfinite_count: int


def max_error(candidate: torch.Tensor, reference: torch.Tensor) -> ErrorStats:
    candidate_finite = torch.isfinite(candidate)
    reference_finite = torch.isfinite(reference)
    finite_mask = candidate_finite & reference_finite

    candidate_nonfinite_count = int((~candidate_finite).sum().item())
    reference_nonfinite_count = int((~reference_finite).sum().item())

    if not finite_mask.any():
        return ErrorStats(
            max_abs=None,
            max_rel=None,
            candidate_nonfinite_count=candidate_nonfinite_count,
            reference_nonfinite_count=reference_nonfinite_count,
        )

    candidate_finite_values = candidate[finite_mask]
    reference_finite_values = reference[finite_mask]
    delta = (candidate_finite_values - reference_finite_values).abs()
    max_abs = float(delta.max().item())

    rel_mask = reference_finite_values.abs() > 1e-5
    if rel_mask.any():
        rel = delta[rel_mask] / reference_finite_values.abs()[rel_mask].clamp_min(1e-5)
        max_rel = float(rel.max().item())
    else:
        max_rel = 0.0

    return ErrorStats(
        max_abs=max_abs,
        max_rel=max_rel,
        candidate_nonfinite_count=candidate_nonfinite_count,
        reference_nonfinite_count=reference_nonfinite_count,
    )


def median_ms(values: Iterable[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.tensor([], dtype=dtype).element_size()


@dataclass(frozen=True)
class AttentionShape:
    batch_size: int
    num_heads: int
    query_len: int
    key_len: int
    head_dim: int

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (self.batch_size, self.num_heads, self.query_len, self.head_dim)
