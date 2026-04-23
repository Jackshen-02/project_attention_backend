from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch


def _normalize_valid_lens(batch_size: int, append_len: int, valid_lens: Iterable[int] | None) -> list[int]:
    if valid_lens is None:
        return [append_len] * batch_size
    lengths = [int(length) for length in valid_lens]
    if len(lengths) != batch_size:
        raise ValueError(f"Expected {batch_size} valid lengths, got {len(lengths)}.")
    for length in lengths:
        if length < 0 or length > append_len:
            raise ValueError(f"Invalid append length {length}; expected 0 <= length <= {append_len}.")
    return lengths


@dataclass
class ContiguousKVCache:
    keys: torch.Tensor
    values: torch.Tensor
    lengths: torch.Tensor
    max_cache_len: int

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "ContiguousKVCache":
        shape = (batch_size, num_heads, max_cache_len, head_dim)
        return cls(
            keys=torch.zeros(shape, device=device, dtype=dtype),
            values=torch.zeros(shape, device=device, dtype=dtype),
            lengths=torch.zeros(batch_size, device=device, dtype=torch.int64),
            max_cache_len=max_cache_len,
        )

    @property
    def batch_size(self) -> int:
        return int(self.keys.size(0))

    @property
    def num_heads(self) -> int:
        return int(self.keys.size(1))

    @property
    def head_dim(self) -> int:
        return int(self.keys.size(-1))

    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        valid_lens: Iterable[int] | None = None,
    ) -> None:
        if k.shape != v.shape:
            raise ValueError("Key and value shapes must match.")
        if k.dim() != 4:
            raise ValueError("Expected K/V tensors of shape (batch, heads, seq, dim).")
        batch_size, _, append_len, _ = k.shape
        if batch_size != self.batch_size:
            raise ValueError("Batch size mismatch for cache append.")

        append_counts = _normalize_valid_lens(batch_size, append_len, valid_lens)
        lengths_cpu = self.lengths.detach().cpu().tolist()
        for batch_idx, valid_len in enumerate(append_counts):
            if valid_len == 0:
                continue
            start = int(lengths_cpu[batch_idx])
            end = start + valid_len
            if end > self.max_cache_len:
                raise ValueError(
                    f"Contiguous KV cache overflow for sequence {batch_idx}: "
                    f"{end} > max_cache_len={self.max_cache_len}."
                )
            self.keys[batch_idx, :, start:end, :] = k[batch_idx, :, :valid_len, :]
            self.values[batch_idx, :, start:end, :] = v[batch_idx, :, :valid_len, :]
            self.lengths[batch_idx] = end

    def clone(self) -> "ContiguousKVCache":
        return ContiguousKVCache(
            keys=self.keys.clone(),
            values=self.values.clone(),
            lengths=self.lengths.clone(),
            max_cache_len=self.max_cache_len,
        )

    def sequence(self, batch_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        length = int(self.lengths[batch_idx].item())
        return (
            self.keys[batch_idx : batch_idx + 1, :, :length, :],
            self.values[batch_idx : batch_idx + 1, :, :length, :],
        )

    def allocated_bytes(self) -> int:
        return int(self.keys.numel() * self.keys.element_size() + self.values.numel() * self.values.element_size())

    def active_bytes(self) -> int:
        active_tokens = int(self.lengths.sum().item())
        per_token_bytes = self.num_heads * self.head_dim * self.keys.element_size() * 2
        return active_tokens * per_token_bytes

    def utilization(self) -> float:
        allocated = self.allocated_bytes()
        return self.active_bytes() / allocated if allocated else 0.0


@dataclass
class PagedKVCache:
    keys: torch.Tensor
    values: torch.Tensor
    lengths: torch.Tensor
    page_size: int
    page_tables: list[list[int]]
    free_pages: list[int]
    used_pages: set[int] = field(default_factory=set)

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        max_pages: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PagedKVCache":
        if page_size <= 0:
            raise ValueError("page_size must be positive.")
        if max_pages <= 0:
            raise ValueError("max_pages must be positive.")
        shape = (max_pages, num_heads, page_size, head_dim)
        return cls(
            keys=torch.zeros(shape, device=device, dtype=dtype),
            values=torch.zeros(shape, device=device, dtype=dtype),
            lengths=torch.zeros(batch_size, device=device, dtype=torch.int64),
            page_size=page_size,
            page_tables=[[] for _ in range(batch_size)],
            free_pages=list(range(max_pages - 1, -1, -1)),
        )

    @property
    def batch_size(self) -> int:
        return int(self.lengths.numel())

    @property
    def num_heads(self) -> int:
        return int(self.keys.size(1))

    @property
    def head_dim(self) -> int:
        return int(self.keys.size(-1))

    @property
    def max_pages(self) -> int:
        return int(self.keys.size(0))

    def _allocate_page(self) -> int:
        if not self.free_pages:
            raise ValueError("Paged KV cache ran out of free pages.")
        page_id = self.free_pages.pop()
        self.used_pages.add(page_id)
        return page_id

    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        valid_lens: Iterable[int] | None = None,
    ) -> None:
        if k.shape != v.shape:
            raise ValueError("Key and value shapes must match.")
        if k.dim() != 4:
            raise ValueError("Expected K/V tensors of shape (batch, heads, seq, dim).")
        batch_size, _, append_len, _ = k.shape
        if batch_size != self.batch_size:
            raise ValueError("Batch size mismatch for cache append.")

        append_counts = _normalize_valid_lens(batch_size, append_len, valid_lens)
        lengths_cpu = self.lengths.detach().cpu().tolist()
        for batch_idx, valid_len in enumerate(append_counts):
            for token_offset in range(valid_len):
                logical_pos = int(lengths_cpu[batch_idx]) + token_offset
                logical_page = logical_pos // self.page_size
                page_offset = logical_pos % self.page_size
                if logical_page == len(self.page_tables[batch_idx]):
                    self.page_tables[batch_idx].append(self._allocate_page())
                page_id = self.page_tables[batch_idx][logical_page]
                self.keys[page_id, :, page_offset, :] = k[batch_idx, :, token_offset, :]
                self.values[page_id, :, page_offset, :] = v[batch_idx, :, token_offset, :]
            self.lengths[batch_idx] = int(lengths_cpu[batch_idx]) + valid_len

    def clone(self) -> "PagedKVCache":
        return PagedKVCache(
            keys=self.keys.clone(),
            values=self.values.clone(),
            lengths=self.lengths.clone(),
            page_size=self.page_size,
            page_tables=[list(table) for table in self.page_tables],
            free_pages=list(self.free_pages),
            used_pages=set(self.used_pages),
        )

    def iter_sequence_blocks(self, batch_idx: int):
        total_length = int(self.lengths[batch_idx].item())
        for logical_page, page_id in enumerate(self.page_tables[batch_idx]):
            start = logical_page * self.page_size
            valid_len = min(self.page_size, total_length - start)
            if valid_len <= 0:
                break
            yield (
                self.keys[page_id : page_id + 1, :, :valid_len, :],
                self.values[page_id : page_id + 1, :, :valid_len, :],
            )

    def page_count(self, batch_idx: int) -> int:
        return len(self.page_tables[batch_idx])

    def allocated_bytes(self) -> int:
        per_page_bytes = (
            self.num_heads
            * self.page_size
            * self.head_dim
            * self.keys.element_size()
            * 2
        )
        return len(self.used_pages) * per_page_bytes

    def active_bytes(self) -> int:
        active_tokens = int(self.lengths.sum().item())
        per_token_bytes = self.num_heads * self.head_dim * self.keys.element_size() * 2
        return active_tokens * per_token_bytes

    def utilization(self) -> float:
        allocated = self.allocated_bytes()
        return self.active_bytes() / allocated if allocated else 0.0


@dataclass(frozen=True)
class PagedKVCachePlan:
    page_size: int
    max_pages: int

    def capacity_tokens(self) -> int:
        return self.page_size * self.max_pages
