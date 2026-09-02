"""Participant-facing compatibility wrapper.

Run from the repository root:
    uv run python starter/submission_checker.py path/to/tokenizer.json
or import:
    from starter.submission_checker import check_submission
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.cli import check_main
from competition.validation import ValidationReport, validate_tokenizer


def check_submission(path: str | Path = "tokenizer.json") -> ValidationReport:
    report = validate_tokenizer(path)
    check_main([str(path)])
    return report


if __name__ == "__main__":
    raise SystemExit(check_main(sys.argv[1:]))
