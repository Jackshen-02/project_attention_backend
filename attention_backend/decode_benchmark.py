from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .benchmark import parse_dtype, select_device
from .common import ErrorStats, max_error, median_ms
from .kv_cache import ContiguousKVCache, PagedKVCache
from .minitorch_bridge import (
    build_minitorch_attention_layer,
    build_shared_attention_problem,
    make_minitorch_backend,
    make_minitorch_decode_input,
    populate_cache_from_prefix,
)


@dataclass(frozen=True)
class DecodeBenchmarkConfig:
    initial_lens: tuple[int, ...] = (512, 1024, 2048, 4096)
    decode_steps: int = 64
    num_heads: int = 8
    head_dim: int = 64
    warmup_iters: int = 3
    measure_iters: int = 10
    page_size: int = 128
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"

    @property
    def batch_size(self) -> int:
        return len(self.initial_lens)

    @property
    def max_cache_len(self) -> int:
        return max(self.initial_lens) + self.decode_steps


@dataclass(frozen=True)
class DecodeBenchmarkResult:
    backend: str
    batch_size: int
    initial_lens: tuple[int, ...]
    decode_steps: int
    latency_ms_per_step: float
    tokens_per_second: float
    allocated_cache_bytes: int
    active_cache_bytes: int
    cache_utilization: float
    pages_used: int | None
    max_abs_error: float | None
    max_rel_error: float | None
    output_nonfinite_count: int
    reference_nonfinite_count: int
    device: str
    dtype: str
    page_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _synchronize_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    try:
        import pycuda.driver as cuda  # type: ignore

        cuda.Context.synchronize()
    except Exception:
        pass


def _benchmark_wallclock(run_decode, cache_factory, iters: int, device: torch.device):
    times_ms: list[float] = []
    out = None
    for _ in range(iters):
        cache = cache_factory()
        _synchronize_device(device)
        start = time.perf_counter()
        out = run_decode(cache)
        _synchronize_device(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
    assert out is not None
    return times_ms, out


def _make_prefilled_cache(config: DecodeBenchmarkConfig, backend_name: str, device: torch.device, dtype: torch.dtype):
    batch_size = config.batch_size
    if backend_name == "contiguous":
        return ContiguousKVCache.allocate(
            batch_size=batch_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            max_cache_len=config.max_cache_len,
            device=device,
            dtype=dtype,
        )
    if backend_name == "paged":
        max_pages = sum(math.ceil((length + config.decode_steps) / config.page_size) for length in config.initial_lens)
        return PagedKVCache.allocate(
            batch_size=batch_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            page_size=config.page_size,
            max_pages=max_pages,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unknown decode backend: {backend_name}")


def _rollout_decode(
    *,
    layer,
    shared,
    mt_backend,
    initial_lens: tuple[int, ...],
    decode_steps: int,
    cache_backend: str,
    cache,
):
    outputs: list[torch.Tensor] = []
    for step in range(decode_steps):
        positions = [initial_lens[batch_idx] + step for batch_idx in range(len(initial_lens))]
        x_step = make_minitorch_decode_input(shared, positions, mt_backend)
        out = layer.decode_step(x_step, cache, cache_backend=cache_backend)
        outputs.append(torch.tensor(out.to_numpy(), dtype=torch.float32))
    return torch.stack(outputs, dim=1), cache


def benchmark_decode_backend(
    backend_name: str,
    config: DecodeBenchmarkConfig,
) -> DecodeBenchmarkResult:
    device = select_device(config.device)
    dtype = parse_dtype(config.dtype)
    _, mt_backend = make_minitorch_backend(device)

    shared = build_shared_attention_problem(
        config.batch_size,
        config.max_cache_len,
        config.num_heads,
        config.head_dim,
        seed=config.seed,
    )
    layer = build_minitorch_attention_layer(
        shared,
        num_heads=config.num_heads,
        backend=mt_backend,
        causal=True,
        attention_backend="naive",
    )

    prototype_cache = _make_prefilled_cache(config, backend_name, device, dtype)
    populate_cache_from_prefix(layer, shared, list(config.initial_lens), mt_backend, prototype_cache)

    def run_decode(cache):
        return _rollout_decode(
            layer=layer,
            shared=shared,
            mt_backend=mt_backend,
            initial_lens=config.initial_lens,
            decode_steps=config.decode_steps,
            cache_backend=backend_name,
            cache=cache,
        )

    for _ in range(config.warmup_iters):
        _ = run_decode(prototype_cache.clone())

    times_ms, (outputs, final_cache) = _benchmark_wallclock(
        run_decode,
        prototype_cache.clone,
        config.measure_iters,
        device,
    )
    latency_ms_total = median_ms(times_ms)
    latency_ms_per_step = latency_ms_total / config.decode_steps
    tokens_per_second = config.batch_size * config.decode_steps / (latency_ms_total / 1000.0)

    if backend_name == "contiguous":
        error_stats = ErrorStats(
            max_abs=0.0,
            max_rel=0.0,
            candidate_nonfinite_count=0,
            reference_nonfinite_count=0,
        )
        pages_used = None
    else:
        # Populate the contiguous reference cache after construction.
        ref_cache = _make_prefilled_cache(config, "contiguous", device, dtype)
        populate_cache_from_prefix(layer, shared, list(config.initial_lens), mt_backend, ref_cache)
        reference_outputs, _ = _rollout_decode(
            layer=layer,
            shared=shared,
            mt_backend=mt_backend,
            initial_lens=config.initial_lens,
            decode_steps=config.decode_steps,
            cache_backend="contiguous",
            cache=ref_cache,
        )
        error_stats = max_error(outputs, reference_outputs)
        pages_used = len(final_cache.used_pages)

    return DecodeBenchmarkResult(
        backend=backend_name,
        batch_size=config.batch_size,
        initial_lens=config.initial_lens,
        decode_steps=config.decode_steps,
        latency_ms_per_step=latency_ms_per_step,
        tokens_per_second=tokens_per_second,
        allocated_cache_bytes=final_cache.allocated_bytes(),
        active_cache_bytes=final_cache.active_bytes(),
        cache_utilization=final_cache.utilization(),
        pages_used=pages_used,
        max_abs_error=error_stats.max_abs,
        max_rel_error=error_stats.max_rel,
        output_nonfinite_count=error_stats.candidate_nonfinite_count,
        reference_nonfinite_count=error_stats.reference_nonfinite_count,
        device=str(device),
        dtype=config.dtype,
        page_size=config.page_size,
    )


def benchmark_decode_suite(config: DecodeBenchmarkConfig) -> list[DecodeBenchmarkResult]:
    results: list[DecodeBenchmarkResult] = []
    backends = ("contiguous", "paged")
    for index, backend_name in enumerate(backends, start=1):
        print(
            f"[{index}/{len(backends)}] decode_backend={backend_name} batch_size={config.batch_size} "
            f"decode_steps={config.decode_steps}",
            flush=True,
        )
        print(
            f"  initial_lens={list(config.initial_lens)} page_size={config.page_size} "
            f"warmup={config.warmup_iters} measure={config.measure_iters}",
            flush=True,
        )
        results.append(benchmark_decode_backend(backend_name, config))
    return results


def decode_results_to_json(results: list[DecodeBenchmarkResult]) -> str:
    return json.dumps([result.to_dict() for result in results], indent=2)
