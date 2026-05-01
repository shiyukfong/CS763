"""Aggregate per-run metric JSONs into a single last-iteration summary.

Input: a folder of per-run JSONs (output of log_to_json.py). Each filename
contains a token identifying the fine-tuning configuration:
    head_only   -> layer = 1
    full        -> layer = 40
    last{x}     -> layer = x

For each metric key in the input files, the final-iteration value is collected
into a list aligned with `layer`. Files are sorted by layer ascending. Output:
    <input-folder>/last_iter_<common-prefix>.json

Usage:
    python aggregate_last_iter.py <path/to/json/folder>
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def extract_layer(stem: str) -> int:
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p == "head" and i + 1 < len(parts) and parts[i + 1] == "only":
            return 1
        if p == "full":
            return 40
        m = re.fullmatch(r"last(\d+)", p)
        if m:
            return int(m.group(1))
    raise ValueError(f"cannot determine layer from filename: {stem}")


def common_prefix(stems: list[str]) -> str:
    prefix = os.path.commonprefix(stems)
    if prefix.endswith("_"):
        prefix = prefix.rstrip("_")
    elif "_" in prefix:
        prefix = prefix.rsplit("_", 1)[0]
    return prefix or "aggregate"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="folder containing per-run JSON files")
    args = parser.parse_args()

    folder: Path = args.folder.resolve()
    if not folder.is_dir():
        print(f"error: not a directory: {folder}", file=sys.stderr)
        return 1

    json_files = sorted(p for p in folder.glob("*.json") if not p.name.startswith("last_iter_"))
    if not json_files:
        print(f"error: no input JSON files in {folder}", file=sys.stderr)
        return 1

    entries: list[tuple[int, Path, dict]] = []
    for path in json_files:
        layer = extract_layer(path.stem)
        with path.open() as f:
            data = json.load(f)
        entries.append((layer, path, data))
    entries.sort(key=lambda e: e[0])

    all_keys: list[str] = []
    seen: set[str] = set()
    for _, _, data in entries:
        for k in data:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    output: dict[str, list] = {"layer": [layer for layer, _, _ in entries]}
    for key in all_keys:
        values: list = []
        for _, _, data in entries:
            series = data.get(key)
            values.append(series[-1] if series else None)
        output[key] = values

    prefix = common_prefix([p.stem for _, p, _ in entries])
    out_path = folder / f"last_iter_{prefix}.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=4)

    print(f"wrote {out_path} ({len(entries)} files, layers={output['layer']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
