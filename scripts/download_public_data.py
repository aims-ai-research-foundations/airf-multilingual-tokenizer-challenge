#!/usr/bin/env python3
"""Download the frozen public Hub dataset and export competition CSV files."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.hub_data import PUBLIC_DATASET_REVISION, load_public_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AMTC train/validation data from Hugging Face")
    parser.add_argument("--output-dir", default="data/public")
    parser.add_argument("--revision", default=PUBLIC_DATASET_REVISION)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        dataset = load_public_split(split, revision=args.revision)
        path = output_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["language", "text"])
            writer.writerows(zip(dataset["language"], dataset["text"], strict=True))
        print(f"{split}: {len(dataset):,} rows → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

