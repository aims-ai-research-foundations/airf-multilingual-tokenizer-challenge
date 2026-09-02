#!/usr/bin/env python3
"""Fail closed unless the complete public/hidden AMTC release is internally sound."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from competition.constants import LANGUAGES
from scripts.build_hf_dataset import MAX_CHARACTERS, MIN_CHARACTERS, SOURCE_REVISION, TARGETS


def _resolve_file(release_dir: Path, recorded_path: str) -> Path:
    recorded = Path(recorded_path)
    candidates = (
        recorded,
        release_dir / "data" / recorded.name,
        release_dir.parent.parent / recorded,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"manifest file is missing: {recorded_path}")


def audit(public_dir: Path, hidden_dir: Path) -> dict:
    import pyarrow.parquet as pq

    manifests = {}
    for visibility, release_dir in (("public", public_dir), ("hidden", hidden_dir)):
        path = release_dir / "manifest.json"
        if not path.is_file():
            raise ValueError(f"missing manifest: {path}")
        manifests[visibility] = json.loads(path.read_text(encoding="utf-8"))

    if manifests["public"]["splits"] != ["train", "validation"]:
        raise ValueError("public release must contain only train and validation")
    if manifests["hidden"]["splits"] != ["public_test", "private_test"]:
        raise ValueError("hidden release must contain only public_test and private_test")

    split_counts = {split: Counter() for split in TARGETS}
    article_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    content_hashes: set[str] = set()
    files_checked = 0
    rows_checked = 0
    for visibility, release_dir in (("public", public_dir), ("hidden", hidden_dir)):
        manifest = manifests[visibility]
        if manifest["source_revision"] != SOURCE_REVISION:
            raise ValueError(f"unexpected source revision in {visibility} manifest")
        for details in manifest["files"]:
            path = _resolve_file(release_dir, details["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != details["sha256"]:
                raise ValueError(f"checksum mismatch: {path}")
            table = pq.read_table(path)
            if table.num_rows != details["rows"]:
                raise ValueError(f"row-count mismatch: {path}")
            split = path.stem.rsplit("-", 1)[0]
            if split not in manifest["splits"]:
                raise ValueError(f"{split} is in the wrong release directory")
            rows = table.to_pylist()
            for row in rows:
                language = row["language"]
                if language not in LANGUAGES:
                    raise ValueError(f"unsupported language {language!r}: {path}")
                text = row["text"]
                if not MIN_CHARACTERS <= len(text) <= MAX_CHARACTERS:
                    raise ValueError(f"invalid passage length in {path}")
                if row["source_revision"] != SOURCE_REVISION:
                    raise ValueError(f"row has unexpected source revision: {path}")
                content_hash = row["content_sha256"]
                if content_hash in content_hashes:
                    raise ValueError(f"duplicate content across release: {content_hash}")
                content_hashes.add(content_hash)
                article_splits[(language, row["source_id"])].add(split)
                split_counts[split][language] += 1
            files_checked += 1
            rows_checked += table.num_rows

    leaked_articles = [key for key, splits in article_splits.items() if len(splits) > 1]
    if leaked_articles:
        raise ValueError(f"articles cross split boundaries; first examples: {leaked_articles[:5]}")
    for split, target in TARGETS.items():
        actual = split_counts[split]
        if set(actual) != set(LANGUAGES) or any(actual[language] != target for language in LANGUAGES):
            raise ValueError(f"incorrect {split} counts: {dict(actual)}")

    expected_rows = sum(TARGETS.values()) * len(LANGUAGES)
    if rows_checked != expected_rows or len(content_hashes) != expected_rows:
        raise ValueError("global row/hash total does not match the release design")
    return {
        "status": "passed",
        "files_checked": files_checked,
        "rows_checked": rows_checked,
        "unique_content_hashes": len(content_hashes),
        "unique_articles": len(article_splits),
        "counts": {split: dict(split_counts[split]) for split in TARGETS},
        "source_revision": SOURCE_REVISION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the complete AMTC dataset release")
    parser.add_argument("--public-dir", default="dataset_release/public")
    parser.add_argument("--hidden-dir", default="dataset_release/hidden")
    parser.add_argument("--report", default="dataset_release/audit-report.json")
    args = parser.parse_args()
    report = audit(Path(args.public_dir), Path(args.hidden_dir))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
