from __future__ import annotations

from pathlib import Path, PurePosixPath

from .submissions import validate_submission_directory
from .validation import validate_tokenizer


def submission_slug_from_changes(changed_files: list[str]) -> str:
    """Enforce one isolated team directory in a competition-entry PR."""
    if not changed_files:
        raise ValueError("pull request contains no changed files")
    slugs = set()
    allowed_names = {"tokenizer.json", "metadata.yml", "README.md"}
    for raw_path in changed_files:
        path = PurePosixPath(raw_path)
        if len(path.parts) != 3 or path.parts[0] != "submissions":
            raise ValueError(f"submission PR may not change {raw_path}")
        if path.name not in allowed_names:
            raise ValueError(f"unexpected submission file: {raw_path}")
        slugs.add(path.parts[1])
    if len(slugs) != 1:
        raise ValueError("a pull request must change exactly one team directory")
    slug = slugs.pop()
    if slug == "baseline":
        raise ValueError("participants may not modify the official baseline")
    return slug


def validate_pr(root: str | Path, changed_files: list[str]) -> dict:
    root = Path(root)
    slug = submission_slug_from_changes(changed_files)
    metadata, tokenizer_path = validate_submission_directory(root / "submissions" / slug)
    report = validate_tokenizer(tokenizer_path)
    if not report.valid:
        raise ValueError("; ".join(report.errors))
    return {"slug": slug, "team": metadata.team, "vocab_size": report.vocab_size}

