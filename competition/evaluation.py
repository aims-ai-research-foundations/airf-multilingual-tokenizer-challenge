from __future__ import annotations

from pathlib import Path

from .data import load_dataset
from .metrics import guardrail_breaches, score_tokenizer
from .submissions import validate_submission_directory
from .validation import validate_tokenizer


def evaluate_submission(
    submission_dir: str | Path,
    evaluation_data: str | Path,
    *,
    benchmark_repeats: int = 3,
) -> dict:
    """Validate and score one submission against an evaluation split."""
    metadata, tokenizer_path = validate_submission_directory(submission_dir)
    report = validate_tokenizer(tokenizer_path)
    if not report.valid or report.tokenizer is None:
        raise ValueError("invalid tokenizer: " + "; ".join(report.errors))

    examples = load_dataset(evaluation_data)
    result = score_tokenizer(
        report.tokenizer, examples, benchmark_repeats=benchmark_repeats
    )
    breaches = guardrail_breaches(result.fertility)
    if breaches:
        detail = ", ".join(
            f"{language} fertility {result.fertility[language]:.3f}"
            for language in breaches
        )
        raise ValueError(f"context-language guardrail exceeded: {detail}")

    return {
        "slug": Path(submission_dir).name,
        "team": metadata.team,
        "members": list(metadata.members),
        "affiliation": metadata.affiliation,
        "approach": metadata.approach,
        "vocab_size": report.vocab_size,
        **result.as_dict(),
    }
