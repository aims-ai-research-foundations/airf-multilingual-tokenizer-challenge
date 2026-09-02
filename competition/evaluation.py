from __future__ import annotations

from pathlib import Path

from .data import load_dataset
from .metrics import load_baseline, score_tokenizer
from .submissions import validate_submission_directory
from .validation import validate_tokenizer


def evaluate_submission(
    submission_dir: str | Path,
    evaluation_data: str | Path,
    baseline_stats: str | Path,
    *,
    benchmark_repeats: int = 3,
) -> dict:
    metadata, tokenizer_path = validate_submission_directory(submission_dir)
    report = validate_tokenizer(tokenizer_path)
    if not report.valid or report.tokenizer is None:
        raise ValueError("invalid tokenizer: " + "; ".join(report.errors))
    examples = load_dataset(evaluation_data)
    baseline = load_baseline(baseline_stats)
    result = score_tokenizer(
        report.tokenizer,
        examples,
        baseline,
        benchmark_repeats=benchmark_repeats,
    )
    return {
        "slug": Path(submission_dir).name,
        "team": metadata.team,
        "members": list(metadata.members),
        "affiliation": metadata.affiliation,
        "approach": metadata.approach,
        "final": metadata.final,
        "vocab_size": report.vocab_size,
        **result.as_dict(),
    }

