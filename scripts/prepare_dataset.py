#!/usr/bin/env python3
"""Deterministically split one-sentence-per-line language source files.

This is the final mechanical stage, not a substitute for the licensing and
native-speaker review gates in DATASET.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
import unicodedata
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.constants import LANGUAGES


def normalized_key(text: str) -> str:
    normalized = unicodedata.normalize("NFC", " ".join(text.split())).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def prepare(input_dir: Path, output_dir: Path, sizes: dict[str, int], seed: int) -> dict:
    rows_by_split = {split: [] for split in sizes}
    seen_globally: set[str] = set()
    required = sum(sizes.values())
    for language_index, language in enumerate(LANGUAGES):
        source = input_dir / f"{language}.txt"
        if not source.is_file():
            raise ValueError(f"missing source file: {source}")
        clean: list[str] = []
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            text = " ".join(raw_line.split()).strip()
            if not text:
                continue
            key = normalized_key(text)
            if key in seen_globally:
                continue
            seen_globally.add(key)
            clean.append(text)
        if len(clean) < required:
            raise ValueError(f"{language}: need {required:,} unique rows, found {len(clean):,}")
        random.Random(seed + language_index).shuffle(clean)
        offset = 0
        for split, size in sizes.items():
            rows_by_split[split].extend((language, text) for text in clean[offset:offset + size])
            offset += size

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split_index, (split, rows) in enumerate(rows_by_split.items()):
        random.Random(seed + 10_000 + split_index).shuffle(rows)
        path = output_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["language", "text"])
            writer.writerows(rows)
        manifest[split] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "path": str(path),
        }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build balanced AMTC CSV splits from reviewed source lines")
    parser.add_argument("input_dir", help="directory containing en.txt, fr.txt, ha.txt, sw.txt, yo.txt, am.txt")
    parser.add_argument("output_dir")
    parser.add_argument("--train", type=int, default=50_000)
    parser.add_argument("--validation", type=int, default=5_000)
    parser.add_argument("--public-test", type=int, default=5_000)
    parser.add_argument("--private-test", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    sizes = {
        "train": args.train,
        "validation": args.validation,
        "public_test": args.public_test,
        "private_test": args.private_test,
    }
    if any(size < 1 for size in sizes.values()):
        parser.error("all split sizes must be positive")
    manifest = prepare(Path(args.input_dir), Path(args.output_dir), sizes, args.seed)
    for split, details in manifest.items():
        print(f"{split:<13} {details['rows']:>8,} rows  sha256={details['sha256']}")
    print("Store test files securely; do not commit them to the public repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

