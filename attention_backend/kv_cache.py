from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ContiguousKVCache:
    keys: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.keys.append(k)
        self.values.append(v)

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.keys:
            raise ValueError("KV cache is empty.")
        return torch.cat(self.keys, dim=-2), torch.cat(self.values, dim=-2)


@dataclass(frozen=True)
class PagedKVCachePlan:
    page_size: int
    max_pages: int

    def capacity_tokens(self) -> int:
        return self.page_size * self.max_pages
