from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot decode page-size tradeoff with paged latency/utilization and contiguous reference lines."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            Path("results/decode/decode_h100_bs4_p32.json"),
            Path("results/decode/decode_h100_bs4_p64.json"),
            Path("results/decode/decode_h100_bs4_p128.json"),
            Path("results/decode/decode_h100_bs4_p256.json"),
            Path("results/decode/decode_h100_bs4_p512.json"),
        ],
        help="Decode benchmark JSON files for the standard workload.",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("results/decode/decode_h100_bs4_p256.json"),
        help="JSON file whose contiguous result will be used for the reference lines.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/decode/decode_tradeoff_dual_axis.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Decode Page-Size Tradeoff on H100",
        help="Figure title.",
    )
    return parser.parse_args()


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_backend_entry(results: list[dict], backend: str) -> dict:
    for entry in results:
        if entry["backend"] == backend:
            return entry
    raise ValueError(f"Backend {backend!r} not found in results.")


def main() -> None:
    args = parse_args()

    page_sizes: list[int] = []
    paged_latency: list[float] = []
    paged_utilization: list[float] = []

    for path in args.inputs:
        results = load_results(path)
        paged = get_backend_entry(results, "paged")
        page_sizes.append(int(paged["page_size"]))
        paged_latency.append(float(paged["latency_ms_per_step"]))
        paged_utilization.append(float(paged["cache_utilization"]))

    baseline_results = load_results(args.baseline_file)
    contiguous = get_backend_entry(baseline_results, "contiguous")
    contiguous_latency = float(contiguous["latency_ms_per_step"])
    contiguous_utilization = float(contiguous["cache_utilization"])

    ordering = sorted(range(len(page_sizes)), key=page_sizes.__getitem__)
    page_sizes = [page_sizes[index] for index in ordering]
    paged_latency = [paged_latency[index] for index in ordering]
    paged_utilization = [paged_utilization[index] for index in ordering]

    plt.style.use("seaborn-v0_8-white")
    fig, ax1 = plt.subplots(figsize=(8.4, 6.2))
    ax2 = ax1.twinx()

    latency_line = ax1.plot(
        page_sizes,
        paged_latency,
        marker="o",
        color="#1f77b4",
        linewidth=2.0,
        label="Paged Latency",
    )[0]
    baseline_latency_line = ax1.axhline(
        contiguous_latency,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label="Contiguous Latency",
    )

    utilization_line = ax2.plot(
        page_sizes,
        paged_utilization,
        marker="s",
        color="#d62728",
        linewidth=2.0,
        label="Paged Utilization",
    )[0]
    baseline_util_line = ax2.axhline(
        contiguous_utilization,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="Contiguous Utilization",
    )

    latency_max = max(max(paged_latency), contiguous_latency)
    latency_pad = max(2.0, 0.08 * latency_max)
    ax1.set_ylim(0.0, latency_max + latency_pad)

    util_min = min(min(paged_utilization), contiguous_utilization)
    util_max = max(max(paged_utilization), contiguous_utilization)
    util_pad = max(0.02, 0.08 * (util_max - util_min))
    ax2.set_ylim(max(0.0, util_min - util_pad), min(1.05, util_max + util_pad))

    ax1.set_title(args.title, fontsize=14, weight="bold")
    ax1.set_xlabel("Page Size")
    ax1.set_ylabel("Latency per Step (ms)", color="#1f77b4")
    ax2.set_ylabel("Cache Utilization", color="#d62728")

    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_xticks(page_sizes)
    ax1.yaxis.set_major_locator(MultipleLocator(5.0))
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.grid(True, which="major", axis="x", color="#d0d0d0", linewidth=0.8, alpha=0.8)
    ax2.grid(False)

    lines = [latency_line, baseline_latency_line, utilization_line, baseline_util_line]
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
