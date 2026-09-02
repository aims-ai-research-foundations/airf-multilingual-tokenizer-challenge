from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .constants import LANGUAGES


@dataclass(frozen=True)
class Example:
    language: str
    text: str


def load_dataset(path: str | Path, *, require_all_languages: bool = True) -> list[Example]:
    """Load and strictly validate a competition CSV."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Dataset not found: {path}")

    examples: list[Example] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["language", "text"]:
            raise ValueError("Dataset columns must be exactly: language,text")
        for row_number, row in enumerate(reader, start=2):
            language = (row.get("language") or "").strip()
            text = row.get("text") or ""
            if language not in LANGUAGES:
                raise ValueError(f"Row {row_number}: unsupported language {language!r}")
            if not text.strip():
                raise ValueError(f"Row {row_number}: text must not be empty")
            examples.append(Example(language, text))

    if not examples:
        raise ValueError("Dataset must contain at least one row")
    counts = Counter(item.language for item in examples)
    missing = [language for language in LANGUAGES if not counts[language]]
    if require_all_languages and missing:
        raise ValueError(f"Dataset is missing languages: {', '.join(missing)}")
    return examples


def language_counts(examples: list[Example]) -> dict[str, int]:
    counts = Counter(item.language for item in examples)
    return {language: counts[language] for language in LANGUAGES}

