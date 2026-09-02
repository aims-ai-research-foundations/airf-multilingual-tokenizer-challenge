#!/usr/bin/env python3
"""Build the complete deterministic AMTC corpus from a pinned Wikipedia snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.constants import LANGUAGES, LANGUAGE_NAMES

SOURCE_DATASET = "wikimedia/wikipedia"
SOURCE_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SOURCE_DATE = "20231101"
SEED = "amtc-v1-2026"
TARGETS = {
    "train": 40_000,
    "validation": 4_000,
    "public_test": 4_000,
    "private_test": 8_000,
}
MIN_CHARACTERS = 40
TARGET_CHARACTERS = 120
MAX_CHARACTERS = 320
MAX_SEGMENTS_PER_ARTICLE = 64
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?።፧፨])(?:[\"'’”»)]*)\s+|\n+")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ROOT = Path(__file__).resolve().parents[1]
CARD_TEMPLATE = ROOT / "organizer-private" / "CARD_TEMPLATE.md"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", CONTROL.sub(" ", text))
    return " ".join(text.split()).strip()


def _hard_chunks(text: str) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and length + added > MAX_CHARACTERS:
            chunks.append(" ".join(current))
            current, length = [], 0
        if len(word) > MAX_CHARACTERS:
            if current:
                chunks.append(" ".join(current))
                current, length = [], 0
            chunks.extend(word[index:index + MAX_CHARACTERS] for index in range(0, len(word), MAX_CHARACTERS))
        else:
            current.append(word)
            length += added
    if current:
        chunks.append(" ".join(current))
    return chunks


def segment_article(text: str) -> list[str]:
    """Create bounded natural passages while preserving Unicode and punctuation."""
    raw_units = [normalize_text(unit) for unit in SENTENCE_BOUNDARY.split(text)]
    units: list[str] = []
    for unit in raw_units:
        if not unit:
            continue
        units.extend(_hard_chunks(unit) if len(unit) > MAX_CHARACTERS else [unit])

    passages: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if current and len(candidate) > TARGET_CHARACTERS and len(current) >= MIN_CHARACTERS:
            passages.append(current)
            current = unit
        elif current and len(candidate) > MAX_CHARACTERS:
            if len(current) >= MIN_CHARACTERS:
                passages.append(current)
            current = unit
        else:
            current = candidate
    if len(current) >= MIN_CHARACTERS:
        passages.append(current)
    return [passage for passage in passages if _is_usable(passage)][:MAX_SEGMENTS_PER_ARTICLE]


def _is_usable(text: str) -> bool:
    if not MIN_CHARACTERS <= len(text) <= MAX_CHARACTERS:
        return False
    letters = sum(character.isalpha() for character in text)
    return letters >= 30 and letters / len(text) >= 0.45


def content_key(text: str) -> str:
    comparable = unicodedata.normalize("NFKC", text).casefold()
    comparable = " ".join(comparable.split())
    return hashlib.sha256(comparable.encode("utf-8")).hexdigest()


def choose_split(language: str, source_id: str, counts: Counter) -> str:
    unfinished = [split for split, target in TARGETS.items() if counts[split] < target]
    if not unfinished:
        raise RuntimeError("language targets already complete")

    def priority(split: str):
        progress = counts[split] / TARGETS[split]
        tie = hashlib.sha256(f"{SEED}|{language}|{source_id}|{split}".encode()).hexdigest()
        return progress, tie

    return min(unfinished, key=priority)


def collect_language(language: str, seen: set[str]) -> tuple[dict[str, list[dict]], dict]:
    from datasets import load_dataset

    config = f"{SOURCE_DATE}.{language}"
    stream = load_dataset(
        SOURCE_DATASET,
        config,
        split="train",
        streaming=True,
        revision=SOURCE_REVISION,
    )
    rows = {split: [] for split in TARGETS}
    counts: Counter = Counter()
    articles_seen = 0
    duplicates_removed = 0
    rejected_segments = 0
    for article in stream:
        articles_seen += 1
        source_id = str(article["id"])
        segments = segment_article(article.get("text") or "")
        if not segments:
            continue
        split = choose_split(language, source_id, counts)
        remaining = TARGETS[split] - counts[split]
        for segment_index, text in enumerate(segments[:remaining]):
            key = content_key(text)
            if key in seen:
                duplicates_removed += 1
                continue
            seen.add(key)
            rows[split].append({
                "language": language,
                "text": text,
                "source_id": source_id,
                "source_url": str(article.get("url") or ""),
                "source_config": config,
                "source_revision": SOURCE_REVISION,
                "segment_index": segment_index,
                "content_sha256": key,
            })
            counts[split] += 1
        rejected_segments += max(0, len(segments) - remaining)
        if articles_seen % 1_000 == 0:
            progress = " ".join(f"{name}={counts[name]:,}/{target:,}" for name, target in TARGETS.items())
            print(f"[{language}] articles={articles_seen:,} {progress}", flush=True)
        if all(counts[split] == target for split, target in TARGETS.items()):
            break

    missing = {split: TARGETS[split] - counts[split] for split in TARGETS if counts[split] < TARGETS[split]}
    if missing:
        raise RuntimeError(f"{language} source exhausted before targets were met: {missing}")
    for split in rows:
        rows[split].sort(key=lambda row: row["content_sha256"])
    return rows, {
        "language": language,
        "language_name": LANGUAGE_NAMES[language],
        "articles_scanned": articles_seen,
        "duplicates_removed": duplicates_removed,
        "segments_discarded_after_quota": rejected_segments,
        "counts": dict(counts),
    }


def write_parquet(rows: list[dict], path: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("language", pa.string()),
        ("text", pa.string()),
        ("source_id", pa.string()),
        ("source_url", pa.string()),
        ("source_config", pa.string()),
        ("source_revision", pa.string()),
        ("segment_index", pa.int32()),
        ("content_sha256", pa.string()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", row_group_size=10_000)
    return {
        "path": str(path),
        "rows": table.num_rows,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build(public_dir: Path, hidden_dir: Path) -> tuple[dict, dict]:
    if public_dir.resolve() == hidden_dir.resolve():
        raise ValueError("public and hidden output directories must be different")
    if not CARD_TEMPLATE.is_file():
        raise FileNotFoundError(f"missing dataset card template: {CARD_TEMPLATE}")
    public_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CARD_TEMPLATE, public_dir / "README.md")
    seen: set[str] = set()
    public_files = []
    hidden_files = []
    language_reports = []
    for language in LANGUAGES:
        rows, report = collect_language(language, seen)
        language_reports.append(report)
        for split, split_rows in rows.items():
            destination_root = public_dir if split in {"train", "validation"} else hidden_dir
            details = write_parquet(split_rows, destination_root / "data" / f"{split}-{language}.parquet")
            (public_files if destination_root == public_dir else hidden_files).append(details)

    common = {
        "dataset_version": "1.0.0",
        "pipeline_version": "3.0.0",
        "seed": SEED,
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "source_date": SOURCE_DATE,
        "targets_per_language": TARGETS,
        "languages": list(LANGUAGES),
        "segmentation": {
            "min_characters": MIN_CHARACTERS,
            "target_characters": TARGET_CHARACTERS,
            "max_characters": MAX_CHARACTERS,
            "max_segments_per_article": MAX_SEGMENTS_PER_ARTICLE,
            "unicode_normalization": "NFC",
        },
        "language_reports": language_reports,
        "global_unique_content_hashes": len(seen),
    }
    public_manifest = {**common, "visibility": "public", "splits": ["train", "validation"], "files": public_files}
    hidden_manifest = {**common, "visibility": "organizer-only", "splits": ["public_test", "private_test"], "files": hidden_files}
    (public_dir / "manifest.json").write_text(json.dumps(public_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (hidden_dir / "manifest.json").write_text(json.dumps(hidden_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return public_manifest, hidden_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build full public and hidden AMTC datasets")
    parser.add_argument("--public-dir", default="dataset_release/public")
    parser.add_argument("--hidden-dir", default="dataset_release/hidden")
    args = parser.parse_args()
    public_manifest, hidden_manifest = build(Path(args.public_dir), Path(args.hidden_dir))
    print(f"Public rows: {sum(file['rows'] for file in public_manifest['files']):,}")
    print(f"Hidden rows: {sum(file['rows'] for file in hidden_manifest['files']):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
