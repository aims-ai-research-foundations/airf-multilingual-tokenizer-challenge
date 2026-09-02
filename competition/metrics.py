from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from .constants import LANGUAGES
from .data import Example


def count_characters(text: str) -> int:
    """Count Unicode code points, excluding Unicode whitespace."""
    return sum(1 for character in text if not character.isspace())


@dataclass(frozen=True)
class ScoreResult:
    score: float
    fertility: dict[str, float]
    normalized: dict[str, float]
    token_counts: dict[str, int]
    character_counts: dict[str, int]
    throughput: float
    elapsed_seconds: float

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "fertility": self.fertility,
            "normalized": self.normalized,
            "token_counts": self.token_counts,
            "character_counts": self.character_counts,
            "throughput": self.throughput,
            "elapsed_seconds": self.elapsed_seconds,
        }


def load_baseline(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fertility = payload.get("fertility", payload)
    missing = [language for language in LANGUAGES if language not in fertility]
    if missing:
        raise ValueError(f"Baseline is missing languages: {', '.join(missing)}")
    values = {language: float(fertility[language]) for language in LANGUAGES}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("All baseline fertility values must be finite and positive")
    return values


def measure_fertility(tokenizer: Tokenizer, examples: list[Example]) -> tuple[dict, dict, dict]:
    token_counts = defaultdict(int)
    character_counts = defaultdict(int)
    texts = [item.text for item in examples]
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    for item, encoding in zip(examples, encodings, strict=True):
        characters = count_characters(item.text)
        if characters == 0:
            raise ValueError("Evaluation rows must contain a non-whitespace character")
        token_counts[item.language] += len(encoding.ids)
        character_counts[item.language] += characters

    fertility = {
        language: token_counts[language] / character_counts[language]
        for language in LANGUAGES
    }
    return fertility, dict(token_counts), dict(character_counts)


def benchmark(tokenizer: Tokenizer, examples: list[Example], *, repeats: int = 3) -> tuple[float, float]:
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
    baseline_fertility: dict[str, float],
    *,
    benchmark_repeats: int = 3,
) -> ScoreResult:
    fertility, token_counts, character_counts = measure_fertility(tokenizer, examples)
    normalized = {
        language: fertility[language] / baseline_fertility[language]
        for language in LANGUAGES
    }
    score = sum(normalized.values()) / len(LANGUAGES)
    throughput, elapsed = benchmark(tokenizer, examples, repeats=benchmark_repeats)
    return ScoreResult(
        score=score,
        fertility=fertility,
        normalized=normalized,
        token_counts=token_counts,
        character_counts=character_counts,
        throughput=throughput,
        elapsed_seconds=elapsed,
    )

