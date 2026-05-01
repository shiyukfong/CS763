#!/usr/bin/env python3
"""Plot per-iteration metric trajectories across multiple runs.

Input: one or more per-run JSON files (output of log_to_json.py). Each file
contains ordered metric lists (index = update_step order).

Output: 7 PNG line plots — one per metric key (acc1_ema, acc5_ema, loss_ema,
nc1-4 EMA). Each line in a plot is one run; the legend shows the number of
trained layers (e.g. "1 (head_only)", "4 (last_4)", "40 (full)"). Runs missing
a given metric are skipped for that plot.

Usage:
    python plot_iterwise_metrics.py <run1.json> [<run2.json> ...] [--output_dir DIR]
    python plot_iterwise_metrics.py <folder>   # all *.json in folder (excl. last_iter_*)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from aggregate_last_iter import extract_layer


METRIC_KEYS = [
    "acc1_ema",
    "acc5_ema",
    "loss_ema",
    "nc1_within_class_collapse_ratio_ema",
    "nc2_mean_simplex_etf_error_ema",
    "nc3_weight_mean_alignment_ema",
    "nc4_self_duality_gap_ema",
]


def label_for(stem: str, layer: int) -> str:
    if "head_only" in stem:
        suffix = "head_only"
    elif layer == 40 and "full" in stem.split("_"):
        suffix = "full"
    else:
        suffix = f"last_{layer}"
    return f"{layer} ({suffix})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Per-run JSON files, or a folder of them (excludes last_iter_*.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: parent of first input).",
    )
    args = parser.parse_args()

    json_paths: list[Path] = []
    for p in args.inputs:
        if p.is_dir():
            json_paths.extend(
                sorted(q for q in p.glob("*.json") if not q.name.startswith("last_iter_"))
            )
        else:
            json_paths.append(p)
    if not json_paths:
        raise SystemExit(f"no input JSON files found in: {args.inputs}")

    runs = []
    for path in json_paths:
        with path.open() as f:
            data = json.load(f)
        layer = extract_layer(path.stem)
        runs.append((layer, path.stem, data))
    runs.sort(key=lambda r: r[0])

    output_dir = args.output_dir or json_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("viridis")
    n_runs = len(runs)
    runs_with_color = [
        (layer, stem, data, cmap(i / max(n_runs - 1, 1)))
        for i, (layer, stem, data) in enumerate(runs)
    ]

    for key in METRIC_KEYS:
        present = [(layer, stem, data, color) for layer, stem, data, color in runs_with_color if key in data]
        if not present:
            print(f"no runs have {key}; skipping")
            continue

        plt.figure(figsize=(6, 4))
        for layer, stem, data, color in present:
            series = data[key]
            iters = list(range(len(series)))
            plt.plot(
                iters,
                series,
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=color,
                label=label_for(stem, layer),
            )

        plt.xlabel("Iteration")
        plt.ylabel(key)
        plt.title(f"{key} vs iteration")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(title="Layers trained", fontsize="small", loc="best")
        plt.tight_layout()
        out_path = output_dir / f"iter_{key}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()