#!/usr/bin/env python3
"""Download one file from the private hidden-evaluation dataset.

Kept separate from the workflow so the auth path is testable. Never prints the
text it downloads, only the row count, so hidden examples cannot leak into CI
logs.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.constants import LANGUAGES


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a hidden evaluation split")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    source = hf_hub_download(args.repo_id, args.file, repo_type="dataset")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(Path(source).read_bytes())

    with destination.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["language", "text"]:
            raise ValueError("hidden data columns must be exactly: language,text")
        counts = {language: 0 for language in LANGUAGES}
        for row in reader:
            language = (row.get("language") or "").strip()
            if language not in counts:
                raise ValueError(f"unsupported language in hidden data: {language!r}")
            counts[language] += 1

    if len(set(counts.values())) != 1 or not all(counts.values()):
        raise ValueError(f"hidden data must be balanced across languages: {counts}")
    print(f"hidden split: {sum(counts.values()):,} rows, "
          f"{next(iter(counts.values())):,} per language")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
