from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from .constants import (
    CONTEXT_FERTILITY_RATIO,
    CONTEXT_LANGUAGES,
    LANGUAGES,
    SCORED_LANGUAGES,
    UNKNOWN_PENALTY,
)
from .data import Example


def count_words(text: str) -> int:
    """Count whitespace-separated words in the original text."""
    return len(text.split())


@dataclass(frozen=True)
class ScoreResult:
    score: float
    fertility: dict[str, float]
    unknown_rate: dict[str, float]
    penalised: dict[str, float]
    token_counts: dict[str, int]
    word_counts: dict[str, int]
    unknown_tokens: int
    throughput: float
    elapsed_seconds: float

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "fertility": self.fertility,
            "unknown_rate": self.unknown_rate,
            "penalised": self.penalised,
            "token_counts": self.token_counts,
            "word_counts": self.word_counts,
            "unknown_tokens": self.unknown_tokens,
            "throughput": self.throughput,
            "elapsed_seconds": self.elapsed_seconds,
        }


def unknown_token_id(tokenizer: Tokenizer) -> int | None:
    """Return the id a tokenizer emits for text it cannot represent."""
    model = json.loads(tokenizer.to_str()).get("model", {})
    name = model.get("unk_token")
    if isinstance(name, str):
        return tokenizer.token_to_id(name)
    unk_id = model.get("unk_id")
    return int(unk_id) if unk_id is not None else None


def measure_fertility(
    tokenizer: Tokenizer, examples: list[Example]
) -> tuple[dict, dict, dict, dict]:
    """Return tokens per word and unknown tokens per word, by language."""
    token_counts: defaultdict = defaultdict(int)
    word_counts: defaultdict = defaultdict(int)
    unknown_counts: defaultdict = defaultdict(int)
    unknown_id = unknown_token_id(tokenizer)

    texts = [item.text for item in examples]
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    for item, encoding in zip(examples, encodings, strict=True):
        words = count_words(item.text)
        if words == 0:
            raise ValueError("Evaluation rows must contain at least one word")
        token_counts[item.language] += len(encoding.ids)
        word_counts[item.language] += words
        if unknown_id is not None:
            unknown_counts[item.language] += sum(
                1 for value in encoding.ids if value == unknown_id
            )

    fertility = {
        language: token_counts[language] / word_counts[language]
        for language in LANGUAGES
        if word_counts[language]
    }
    unknown_rate = {
        language: unknown_counts[language] / word_counts[language]
        for language in LANGUAGES
        if word_counts[language]
    }
    return fertility, unknown_rate, dict(token_counts), dict(word_counts)


def penalised_scores(
    fertility: dict[str, float], unknown_rate: dict[str, float]
) -> dict[str, float]:
    """Combine fertility with the unknown-token penalty for each language."""
    return {
        language: value + UNKNOWN_PENALTY * unknown_rate.get(language, 0.0)
        for language, value in fertility.items()
    }


def competition_score(
    fertility: dict[str, float], unknown_rate: dict[str, float] | None = None
) -> float:
    """Average penalised fertility across the four scored languages."""
    missing = [language for language in SCORED_LANGUAGES if language not in fertility]
    if missing:
        raise ValueError(f"missing scored languages: {', '.join(missing)}")
    scores = penalised_scores(fertility, unknown_rate or {})
    return sum(scores[language] for language in SCORED_LANGUAGES) / len(SCORED_LANGUAGES)


def guardrail_breaches(fertility: dict[str, float]) -> list[str]:
    """Return context languages that cost too much relative to the scored ones.

    The limit is relative to the submission's own scored average, so a
    deliberately simple tokenizer is judged on balance rather than on absolute
    fertility.
    """
    budget = competition_score(fertility, {}) * CONTEXT_FERTILITY_RATIO
    return [
        language
        for language in CONTEXT_LANGUAGES
        if fertility.get(language, 0.0) > budget
    ]


def round_trip_failures(
    tokenizer: Tokenizer, examples: list[Example], *, limit: int = 5
) -> list[str]:
    """Return evaluation texts a tokenizer cannot reproduce exactly."""
    texts = [item.text for item in examples]
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    failures = []
    for text, encoding in zip(texts, encodings, strict=True):
        if tokenizer.decode(encoding.ids, skip_special_tokens=False) != text:
            failures.append(text)
            if len(failures) >= limit:
                break
    return failures


def benchmark(
    tokenizer: Tokenizer, examples: list[Example], *, repeats: int = 3
) -> tuple[float, float]:
    """Return characters per second and the median elapsed seconds."""
    texts = [item.text for item in examples]
    total_characters = sum(len(text) for text in texts)
    tokenizer.encode_batch(texts, add_special_tokens=False)  # warm-up
    timings = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        tokenizer.encode_batch(texts, add_special_tokens=False)
        timings.append(time.perf_counter() - started)
    elapsed = statistics.median(timings)
    return total_characters / max(elapsed, 1e-12), elapsed


def score_tokenizer(
    tokenizer: Tokenizer,
    examples: list[Example],
    *,
    benchmark_repeats: int = 3,
) -> ScoreResult:
    """Score a tokenizer with the official competition metric."""
    fertility, unknown_rate, token_counts, word_counts = measure_fertility(
        tokenizer, examples
    )
    throughput, elapsed = benchmark(tokenizer, examples, repeats=benchmark_repeats)
    return ScoreResult(
        score=competition_score(fertility, unknown_rate),
        fertility=fertility,
        unknown_rate=unknown_rate,
        penalised=penalised_scores(fertility, unknown_rate),
        token_counts=token_counts,
        word_counts=word_counts,
        unknown_tokens=sum(
            round(rate * word_counts[language])
            for language, rate in unknown_rate.items()
        ),
        throughput=throughput,
        elapsed_seconds=elapsed,
    )


def load_baseline(path: str | Path) -> dict[str, float]:
    """Load reference fertility values, kept for reporting comparisons."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        language: float(value)
        for language, value in payload.get("fertility", payload).items()
    }
