#!/usr/bin/env python3
"""Plot layerwise finetuning metrics from an aggregated last-iter JSON.

Input: JSON produced by aggregate_last_iter.py with shape:
    {
      "layer": [1, 2, 4, ..., 40],
      "acc1_ema": [...],
      "acc5_ema": [...],
      "loss_ema": [...],
      "nc1_within_class_collapse_ratio_ema": [...],   # may be missing or contain nulls
      "nc2_mean_simplex_etf_error_ema": [...],
      "nc3_weight_mean_alignment_ema": [...],
      "nc4_self_duality_gap_ema": [...]
    }

Outputs:
    <stem>_acc.png  -- line plot of acc1_ema and acc5_ema vs layer count.
    <stem>_nc.png   -- grouped bar plot of NC1-4 metrics vs layer count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NC_METRICS = [
    (
        "nc1_within_class_collapse_ratio_ema",
        "Within-class collapse ratio (lower is better)",
    ),
    (
        "nc2_mean_simplex_etf_error_ema",
        "Mean simplex ETF error (lower is better)",
    ),
    (
        "nc3_weight_mean_alignment_ema",
        "Weight-mean alignment (higher is better)",
    ),
    (
        "nc4_self_duality_gap_ema",
        "Self-duality gap (lower is better)",
    ),
]


def plot_accuracy(layers: list[int], acc1: list, acc5: list, output_path: Path) -> None:
    """Line plot of acc1_ema and acc5_ema. Skips null entries per series."""
    pairs1 = [(l, v) for l, v in zip(layers, acc1) if v is not None]
    pairs5 = [(l, v) for l, v in zip(layers, acc5) if v is not None]

    plt.figure(figsize=(5, 4))
    if pairs1:
        plt.plot(*zip(*pairs1), marker="o", linewidth=2, label="acc1")
    if pairs5:
        plt.plot(*zip(*pairs5), marker="s", linewidth=2, label="acc5")
    plt.xlabel("Number of trained layers")
    plt.ylabel("Accuracy")
    plt.title("Layerwise Finetuning Accuracy")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_nc(layers: list[int], data: dict, output_path: Path) -> None:
    """Grouped bar plot of NC1-4. Drops layers with all-null NC values; omits
    individual bars for null entries (gap in the group)."""
    available = [(k, label) for k, label in NC_METRICS if k in data]
    if not available:
        print("no NC metrics in input; skipping NC plot")
        return

    keep_idx = [
        i for i in range(len(layers))
        if any(data[k][i] is not None for k, _ in available)
    ]
    if not keep_idx:
        print("all NC values are null; skipping NC plot")
        return

    kept_layers = [layers[i] for i in keep_idx]
    x = np.arange(len(keep_idx))
    n = len(available)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(6.0, len(keep_idx) * 0.9), 4.5))
    for j, (key, label) in enumerate(available):
        offsets, heights = [], []
        for xi, src_i in zip(x, keep_idx):
            v = data[key][src_i]
            if v is not None:
                offsets.append(xi + (j - (n - 1) / 2) * width)
                heights.append(v)
        if offsets:
            ax.bar(offsets, heights, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in kept_layers])
    ax.set_xlabel("Number of trained layers")
    ax.set_ylabel("Metric value")
    ax.set_title("Neural Collapse Metrics")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a last_iter_*.json file produced by aggregate_last_iter.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to store generated plots. Defaults to the input file's directory.",
    )
    args = parser.parse_args()

    with args.input.open() as f:
        data = json.load(f)
    layers = data["layer"]

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    acc_path = output_dir / f"{stem}_acc.png"
    nc_path = output_dir / f"{stem}_nc.png"

    plot_accuracy(layers, data.get("acc1_ema", []), data.get("acc5_ema", []), acc_path)
    plot_nc(layers, data, nc_path)

    print(f"Saved {acc_path}")
    print(f"Saved {nc_path}")


if __name__ == "__main__":
    main()