#!/usr/bin/env python3
"""Validate and publish the public train/validation release to Hugging Face."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.constants import LANGUAGES
from competition.hub_data import PUBLIC_DATASET_ID, PUBLIC_DATASET_REVISION


def validate_release(path: Path) -> dict:
    import pyarrow.parquet as pq

    manifest_path = path / "manifest.json"
    card_path = path / "README.md"
    if not manifest_path.is_file() or not card_path.is_file():
        raise ValueError("release requires manifest.json and README.md")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["splits"] != ["train", "validation"]:
        raise ValueError("only train and validation may be published")
    expected = {"train": 40_000, "validation": 4_000}
    totals = {split: {language: 0 for language in LANGUAGES} for split in expected}
    for details in manifest["files"]:
        file_path = Path(details["path"])
        if not file_path.is_absolute():
            file_path = path.parent.parent / file_path
        if not file_path.is_file():
            candidate = path / "data" / Path(details["path"]).name
            file_path = candidate
        checksum = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if checksum != details["sha256"]:
            raise ValueError(f"checksum mismatch: {file_path}")
        table = pq.read_table(file_path, columns=["language"])
        values = table.column("language").to_pylist()
        split = file_path.stem.rsplit("-", 1)[0]
        for language in values:
            totals[split][language] += 1
    for split, per_language in totals.items():
        if any(count != expected[split] for count in per_language.values()):
            raise ValueError(f"incorrect {split} counts: {per_language}")
    return manifest


def main() -> int:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description="Publish the validated public AMTC dataset")
    parser.add_argument("--release-dir", default="dataset_release/public")
    parser.add_argument("--repo-id", default=PUBLIC_DATASET_ID)
    parser.add_argument("--tag", default=PUBLIC_DATASET_REVISION)
    parser.add_argument("--private", action="store_true", help="create a private staging repository")
    args = parser.parse_args()
    release_dir = Path(args.release_dir)
    manifest = validate_release(release_dir)
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=release_dir,
        commit_message=f"Publish AMTC dataset v{manifest['dataset_version']}",
    )
    api.create_tag(args.repo_id, tag=args.tag, repo_type="dataset", tag_message="Frozen competition data v1.0.0")
    print(f"Published commit: {commit.oid}")
    print(f"Dataset: https://huggingface.co/datasets/{args.repo_id}/tree/{args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

