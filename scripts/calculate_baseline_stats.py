#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.constants import LANGUAGES
from competition.data import load_dataset
from competition.metrics import measure_fertility


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate split-specific AMTC baseline fertility")
    parser.add_argument("--tokenizer", default="submissions/baseline/tokenizer.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    fertility, tokens, characters = measure_fertility(tokenizer, load_dataset(args.data))
    payload = {
        "metric_version": "1.0",
        "character_definition": "Unicode code points excluding whitespace",
        "evaluation_data": Path(args.data).name,
        "fertility": {language: fertility[language] for language in LANGUAGES},
        "token_counts": tokens,
        "character_counts": characters,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved baseline statistics for {args.data} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

