from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from .constants import LANGUAGES, MAX_TIME_MULTIPLE, SCORED_LANGUAGES
from .evaluation import evaluate_submission

FIELDS = [
    "rank", "team", "slug", "score", "fertility", "unknown_rate",
    "speed_chars_per_second", "vocab_size", "status", "evaluated_at",
    *[f"{language}_fertility" for language in LANGUAGES],
]

PUBLIC_REPOSITORY = "aims-ai-research-foundations/airf-multilingual-tokenizer-challenge"

METRIC_SUMMARY = (
    "Average tokens per word plus a 100x unknown-token penalty, across Hausa, "
    "Swahili, Yoruba and Amharic. Lower is better."
)


def _scored_mean(values: dict[str, float]) -> float:
    """Return the mean of a per-language measure over the scored languages."""
    return sum(values[language] for language in SCORED_LANGUAGES) / len(SCORED_LANGUAGES)


def build_leaderboard(
    submissions_dir: str | Path,
    evaluation_data: str | Path,
    *,
    benchmark_repeats: int = 3,
    time_limit_multiple: float = MAX_TIME_MULTIPLE,
) -> tuple[list[dict], list[dict]]:
    submissions_dir = Path(submissions_dir)
    rows: list[dict] = []
    failures: list[dict] = []
    evaluated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    baseline_seconds = None
    for directory in sorted(submissions_dir.iterdir() if submissions_dir.exists() else []):
        if not directory.is_dir():
            continue
        try:
            result = evaluate_submission(
                directory,
                evaluation_data,
                benchmark_repeats=benchmark_repeats,
            )
            if directory.name == "baseline":
                baseline_seconds = result["elapsed_seconds"]
            elif baseline_seconds is not None:
                budget = baseline_seconds * time_limit_multiple
                if result["elapsed_seconds"] > budget:
                    raise ValueError(
                        f"evaluation took {result['elapsed_seconds']:.2f}s, over the "
                        f"{time_limit_multiple:g}x baseline limit of {budget:.2f}s"
                    )
            rows.append({
                "team": result["team"],
                "slug": result["slug"],
                "score": result["score"],
                "fertility": _scored_mean(result["fertility"]),
                "unknown_rate": _scored_mean(result["unknown_rate"]),
                "speed_chars_per_second": result["throughput"],
                "vocab_size": result["vocab_size"],
                "status": "baseline" if result["slug"] == "baseline" else "ranked",
                "evaluated_at": evaluated_at,
                **{
                    f"{language}_fertility": result["fertility"][language]
                    for language in LANGUAGES
                },
            })
        except Exception as exc:
            failures.append({"slug": directory.name, "error": str(exc)})

    # Ties on score are broken by throughput: the fastest tokenizer wins.
    competitors = sorted(
        (row for row in rows if row["status"] == "ranked"),
        key=lambda row: (row["score"], -row["speed_chars_per_second"], row["team"].casefold()),
    )
    for rank, row in enumerate(competitors, start=1):
        row["rank"] = rank
    baselines = [row for row in rows if row["status"] == "baseline"]
    for row in baselines:
        row["rank"] = "—"
    return competitors + baselines, failures


def write_leaderboard(
    rows: list[dict],
    csv_path: str | Path,
    markdown_path: str | Path,
    *,
    json_path: str | Path | None = None,
) -> None:
    """Write the leaderboard as CSV, Markdown and, optionally, JSON.

    The JSON file is the feed the public website reads, so its shape is a
    published contract. Add fields to it rather than renaming existing ones.
    """
    csv_path = Path(csv_path)
    markdown_path = Path(markdown_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in FIELDS})

    lines = [
        "# 🏆 Leaderboard",
        "",
        "Score is the average of tokens-per-word plus a 100x unknown-token "
        "penalty, across Hausa, Swahili, Yoruba and Amharic. Lower is "
        "better. Speed is informational.",
        "",
        "| Rank | Team | Score ↓ | Hausa | Swahili | Yoruba | Amharic | Speed ↑ | Vocabulary |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        speed = f"{row['speed_chars_per_second'] / 1_000_000:.2f}M chars/s"
        team = str(row["team"]).replace("|", "\\|")
        label = f"{team} (baseline)" if row["status"] == "baseline" else team
        per_language = " | ".join(
            f"{row[f'{language}_fertility']:.3f}" for language in SCORED_LANGUAGES
        )
        lines.append(
            f"| {row['rank']} | {label} | {row['score']:.4f} | {per_language} | "
            f"{speed} | {row['vocab_size']:,} |"
        )
    lines.extend(["", "Generated by `scripts/update_leaderboard.py`.", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    if json_path is not None:
        write_leaderboard_json(rows, json_path)


def write_leaderboard_json(rows: list[dict], json_path: str | Path) -> None:
    """Write the leaderboard feed consumed by the public website."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    payload = {
        "generated_at": generated_at,
        "metric": METRIC_SUMMARY,
        "repository": PUBLIC_REPOSITORY,
        "entries": [
            {
                "rank": row["rank"],
                "team": row["team"],
                "slug": row["slug"],
                "status": row["status"],
                "score": round(float(row["score"]), 4),
                "fertility": round(float(row["fertility"]), 4),
                "unknown_rate": round(float(row["unknown_rate"]), 6),
                "speed_chars_per_second": int(row["speed_chars_per_second"]),
                "vocab_size": int(row["vocab_size"]),
                "evaluated_at": row["evaluated_at"],
                "fertility_by_language": {
                    language: round(float(row[f"{language}_fertility"]), 4)
                    for language in LANGUAGES
                },
            }
            for row in rows
        ],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _csv_value(value):
    if isinstance(value, float):
        return f"{value:.8f}"
    return value
