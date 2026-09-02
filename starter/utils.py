"""Submission profiler for the AI Research Foundations Multilingual
Tokenization Challenge.

This is the only helper file participants need. Download it next to your
notebook and run it on the ``tokenizer.json`` you are about to submit::

    from utils import profile_submission

    profile_submission("tokenizer.json", data=validation)

The profiler applies the same validity contract as official evaluation,
reports normalized token fertility for every competition language, and
benchmarks throughput. Everything it needs is fetched on demand, so the
file works unchanged in Google Colab, on Kaggle, or in a local clone.
"""

from __future__ import annotations

import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import __version__ as installed_tokenizers_version

__all__ = ["profile_submission"]

GITHUB_REPO = "aims-ai-research-foundations/airf-multilingual-tokenizer-challenge"
GITHUB_BRANCH = "main"
BASELINE_PATH = "submissions/baseline/tokenizer.json"
CACHE_DIR = Path(".amtc_cache")

LANGUAGES = ("en", "fr", "ha", "sw", "yo", "am")
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "ha": "Hausa",
    "sw": "Swahili",
    "yo": "Yoruba",
    "am": "Amharic",
}
MAX_VOCAB_SIZE = 10_000
MAX_TOKENIZER_BYTES = 20 * 1024 * 1024
REQUIRED_TOKENIZERS_VERSION = "0.22.1"
SMOKE_TEXTS = {
    "en": "Knowledge grows when it is shared.",
    "fr": "Le savoir grandit lorsqu’il est partagé.",
    "ha": "Ilimi yana ƙaruwa idan an raba shi.",
    "sw": "Maarifa hukua yanaposhirikishwa.",
    "yo": "Ìmọ̀ ń pọ̀ sí i nígbà tí a bá pín in.",
    "am": "እውቀት ሲካፈል ያድጋል።",
}


def _baseline_url() -> str:
    """Return the raw GitHub URL of the official baseline tokenizer."""
    return (
        f"https://raw.githubusercontent.com/{GITHUB_REPO}/"
        f"{GITHUB_BRANCH}/{BASELINE_PATH}"
    )


def _official_baseline() -> Tokenizer:
    """Load the official baseline tokenizer, downloading it if needed.

    A local competition clone is used when one is available, so the
    function costs nothing after the first call.
    """
    local = Path(BASELINE_PATH)
    if local.is_file():
        return Tokenizer.from_file(str(local))

    cached = CACHE_DIR / "baseline_tokenizer.json"
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_baseline_url(), cached)
    return Tokenizer.from_file(str(cached))


def _rows(data) -> list[tuple[str, str]]:
    """Coerce supported data containers into ``(language, text)`` pairs.

    Accepts a pandas DataFrame, a Hugging Face dataset, a sequence of
    mappings, or a sequence of two-item pairs.
    """
    if hasattr(data, "itertuples"):
        return [(row.language, row.text) for row in data.itertuples()]
    if hasattr(data, "column_names"):
        return list(zip(data["language"], data["text"], strict=True))

    pairs: list[tuple[str, str]] = []
    for item in data:
        if isinstance(item, dict):
            pairs.append((item["language"], item["text"]))
        elif hasattr(item, "language") and hasattr(item, "text"):
            pairs.append((item.language, item.text))
        else:
            language, text = item
            pairs.append((language, text))
    return pairs


def _count_characters(text: str) -> int:
    """Count Unicode code points, excluding every whitespace character."""
    return sum(1 for character in text if not character.isspace())


def _fertility(tokenizer: Tokenizer, rows: list[tuple[str, str]]) -> dict:
    """Return tokens per non-whitespace character for each language."""
    tokens: defaultdict = defaultdict(int)
    characters: defaultdict = defaultdict(int)
    texts = [text for _, text in rows]
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    for (language, text), encoding in zip(rows, encodings, strict=True):
        tokens[language] += len(encoding.ids)
        characters[language] += _count_characters(text)
    return {
        language: tokens[language] / characters[language]
        for language in LANGUAGES
        if characters[language]
    }


def _benchmark(
    tokenizer: Tokenizer,
    rows: list[tuple[str, str]],
    repeats: int = 3,
) -> tuple[float, float]:
    """Return characters per second and the median elapsed seconds."""
    texts = [text for _, text in rows]
    total_characters = sum(len(text) for text in texts)
    tokenizer.encode_batch(texts, add_special_tokens=False)
    timings = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        tokenizer.encode_batch(texts, add_special_tokens=False)
        timings.append(time.perf_counter() - started)
    elapsed = statistics.median(timings)
    return total_characters / max(elapsed, 1e-12), elapsed


def _validate(path: Path) -> tuple[Tokenizer | None, dict, list[str], int | None]:
    """Apply the official validity contract to one tokenizer file."""
    checks: dict[str, bool] = {}
    errors: list[str] = []

    if not path.is_file():
        return None, {"loads": False}, [f"{path} was not found"], None
    if path.name != "tokenizer.json":
        errors.append("the submitted file must be named tokenizer.json")

    checks["file_size"] = path.stat().st_size <= MAX_TOKENIZER_BYTES
    if not checks["file_size"]:
        limit = MAX_TOKENIZER_BYTES // 1_048_576
        errors.append(f"tokenizer.json is larger than {limit} MiB")
        return None, checks, errors, None

    try:
        tokenizer = Tokenizer.from_file(str(path))
        checks["loads"] = True
    except Exception as error:
        checks["loads"] = False
        errors.append(f"tokenizer could not be loaded: {error}")
        return None, checks, errors, None

    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    checks["vocabulary"] = vocab_size <= MAX_VOCAB_SIZE
    if not checks["vocabulary"]:
        errors.append(
            f"vocabulary has {vocab_size:,} entries; the limit is "
            f"{MAX_VOCAB_SIZE:,}"
        )

    try:
        encodings = tokenizer.encode_batch(
            list(SMOKE_TEXTS.values()), add_special_tokens=False
        )
        checks["encoding"] = all(encoding.ids for encoding in encodings)
        if not checks["encoding"]:
            errors.append("at least one language produced no tokens")
        decoded = [
            tokenizer.decode(encoding.ids, skip_special_tokens=False)
            for encoding in encodings
        ]
        checks["decoding"] = all(text.strip() for text in decoded)
        if not checks["decoding"]:
            errors.append("at least one smoke text could not be decoded")
    except Exception as error:
        checks["encoding"] = False
        checks["decoding"] = False
        errors.append(f"encode or decode failed: {error}")

    checks["compatibility"] = (
        installed_tokenizers_version == REQUIRED_TOKENIZERS_VERSION
    )
    if not checks["compatibility"]:
        errors.append(
            f"expected tokenizers {REQUIRED_TOKENIZERS_VERSION} but found "
            f"{installed_tokenizers_version}"
        )
    return tokenizer, checks, errors, vocab_size


def _leader(label: str, width: int = 26) -> str:
    """Format a label padded with dots so report columns line up."""
    return f"{label}{'.' * max(1, width - len(label))}"


def profile_submission(
    path: str | Path = "tokenizer.json",
    data=None,
    *,
    repeats: int = 3,
    verbose: bool = True,
) -> dict:
    """Validate, score and benchmark a tokenizer before you submit it.

    Args:
        path: Location of the ``tokenizer.json`` file to profile.
        data: Optional labelled text used for the competition score and
            the throughput benchmark. Any container of ``language`` and
            ``text`` pairs works, including a pandas DataFrame and a
            Hugging Face dataset. Validation data is the usual choice.
        repeats: Number of timed encoding passes; the median is reported.
        verbose: Print the human readable report as well as returning it.

    Returns:
        A dictionary with the validity checks, any errors, the
        vocabulary size and, when ``data`` is supplied, per language
        fertility, normalized fertility, the competition score and
        throughput.
    """
    path = Path(path)
    tokenizer, checks, errors, vocab_size = _validate(path)
    report = {
        "path": str(path),
        "valid": bool(tokenizer) and all(checks.values()) and not errors,
        "checks": checks,
        "errors": errors,
        "vocab_size": vocab_size,
        "tokenizers_version": installed_tokenizers_version,
    }

    if tokenizer is not None and data is not None:
        rows = _rows(data)
        fertility = _fertility(tokenizer, rows)
        baseline = _fertility(_official_baseline(), rows)
        normalized = {
            language: fertility[language] / baseline[language]
            for language in fertility
        }
        throughput, elapsed = _benchmark(tokenizer, rows, repeats=repeats)
        report.update(
            {
                "fertility": fertility,
                "baseline_fertility": baseline,
                "normalized": normalized,
                "score": sum(normalized.values()) / len(normalized),
                "throughput": throughput,
                "elapsed_seconds": elapsed,
                "rows": len(rows),
            }
        )

    if verbose:
        _print_report(report)
    return report


def _print_report(report: dict) -> None:
    """Print a submission profile in the official checker layout."""
    print("AI Research Foundations Multilingual Tokenization Challenge")
    print("Submission checker")
    print()
    labels = {
        "loads": "Loading tokenizer",
        "file_size": "File size",
        "vocabulary": "Vocabulary",
        "encoding": "Encoding",
        "decoding": "Decoding",
        "compatibility": "Compatibility",
    }
    for key, label in labels.items():
        if key not in report["checks"]:
            continue
        mark = "✓" if report["checks"][key] else "✗"
        detail = ""
        if key == "vocabulary" and report["vocab_size"] is not None:
            detail = f" {report['vocab_size']:,} / {MAX_VOCAB_SIZE:,}"
        print(f"{_leader(label)} {mark}{detail}")

    if "score" in report:
        print()
        print(f"Competition score ({report['rows']:,} rows, lower is better)")
        for language in LANGUAGES:
            if language not in report["normalized"]:
                continue
            print(
                f"{_leader(LANGUAGE_NAMES[language])} "
                f"fertility {report['fertility'][language]:.4f}   "
                f"normalized {report['normalized'][language]:.4f}"
            )
        print(f"{_leader('SCORE')} {report['score']:.4f}")
        print()
        print("Local benchmark (informational only)")
        elapsed = report["elapsed_seconds"]
        readable = f"{elapsed:.2f} s" if elapsed >= 1 else f"{elapsed * 1000:.0f} ms"
        print(f"{_leader('Evaluation time')} {readable}")
        print(
            f"{_leader('Throughput')} "
            f"{report['throughput'] / 1e6:.1f}M characters/sec"
        )

    print()
    print("READY FOR SUBMISSION ✓" if report["valid"] else "NOT READY FOR SUBMISSION")
    for error in report["errors"]:
        print(f"✗ {error}")


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "tokenizer.json"
    profiled = profile_submission(target)
    raise SystemExit(0 if profiled["valid"] else 1)
