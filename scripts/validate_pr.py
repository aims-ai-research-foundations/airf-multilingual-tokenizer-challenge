#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.pr import validate_pr


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the submission changed by a pull request")
    parser.add_argument("--base", required=True, help="base commit SHA")
    parser.add_argument("--head", default="HEAD", help="head commit SHA")
    args = parser.parse_args()
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{args.base}...{args.head}"],
        check=True,
        capture_output=True,
        text=True,
    )
    changed = [line for line in completed.stdout.splitlines() if line.strip()]
    result = validate_pr(Path.cwd(), changed)
    print(f"✓ tokenizer.json found\n✓ Tokenizer loads successfully")
    print(f"✓ Vocabulary: {result['vocab_size']:,} / 10,000")
    print(f"✓ Submission format valid\n\nSUBMISSION ACCEPTED — {result['team']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

