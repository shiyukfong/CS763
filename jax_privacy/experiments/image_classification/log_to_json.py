"""Extract per-step EMA eval metrics from a training log into a JSON file.

Usage:
    python log_to_json.py <path/to/logs/foo.log>
    python log_to_json.py <path/to/logs>           # process all *.log in folder

Output:
    <parent-of-logs>/json/foo.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


TARGET_KEYS = [
    "acc1_ema",
    "acc5_ema",
    "loss_ema",
    "nc1_within_class_collapse_ratio_ema",
    "nc2_mean_simplex_etf_error_ema",
    "nc3_weight_mean_alignment_ema",
    "nc4_self_duality_gap_ema",
]

STEP_RE = re.compile(r"'eval/update_step':\s*(?:array\()?(-?\d+)")
VALUE_RES = {
    k: re.compile(rf"'eval/{k}':\s*array\(\s*([-\d.eE+]+)") for k in TARGET_KEYS
}


def parse_log(log_path: Path) -> dict[str, list[float]]:
    seen_steps: set[int] = set()
    rows: list[tuple[int, dict[str, float]]] = []

    with log_path.open() as f:
        for line in f:
            if "'eval/update_step'" not in line:
                continue
            step_match = STEP_RE.search(line)
            if step_match is None:
                continue
            step = int(step_match.group(1))
            if step in seen_steps:
                continue
            seen_steps.add(step)

            row: dict[str, float] = {}
            for key, regex in VALUE_RES.items():
                m = regex.search(line)
                if m is not None:
                    row[key] = float(m.group(1))
            rows.append((step, row))

    rows.sort(key=lambda x: x[0])

    output: dict[str, list[float]] = {}
    for key in TARGET_KEYS:
        values = [row[key] for _, row in rows if key in row]
        if values:
            output[key] = values
    return output


def process_log(log_path: Path) -> None:
    metrics = parse_log(log_path)
    out_dir = log_path.parent.parent / "json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{log_path.stem}.json"
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=4)
    print(f"wrote {out_path} ({len(metrics)} keys)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a .log file or a folder of .log files")
    args = parser.parse_args()

    path: Path = args.path.resolve()
    if path.is_file():
        process_log(path)
        return 0
    if path.is_dir():
        log_files = sorted(path.glob("*.log"))
        if not log_files:
            print(f"error: no *.log files in {path}", file=sys.stderr)
            return 1
        for log_path in log_files:
            process_log(log_path)
        return 0

    print(f"error: not a file or directory: {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
