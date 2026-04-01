from __future__ import annotations

import argparse
from pathlib import Path

from attention_backend.benchmark import BenchmarkConfig, benchmark_suite, results_to_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark naive vs tiled attention for prefill workloads.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--non-causal", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def format_megabytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 ** 2):.2f}"


def format_error(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3e}"


def main() -> None:
    args = build_parser().parse_args()
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        seq_lens=tuple(args.seq_lens),
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        causal=not args.non_causal,
        block_size=args.block_size,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )

    results = benchmark_suite(config)

    header = (
        f"{'backend':<12} {'seq_len':>8} {'latency_ms':>12} "
        f"{'tok/s':>12} {'peak_mb':>10} {'est_peak_mb':>12} "
        f"{'max_abs_err':>12} {'max_rel_err':>12} "
        f"{'out_nonfinite':>13} {'ref_nonfinite':>13}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.backend:<12} {result.seq_len:>8d} {result.latency_ms:>12.3f} "
            f"{result.tokens_per_second:>12.2f} {format_megabytes(result.peak_memory_bytes):>10} "
            f"{format_megabytes(result.estimated_peak_intermediate_bytes):>12} "
            f"{format_error(result.max_abs_error):>12} {format_error(result.max_rel_error):>12} "
            f"{result.output_nonfinite_count:>13d} {result.reference_nonfinite_count:>13d}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(results_to_json(results) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
