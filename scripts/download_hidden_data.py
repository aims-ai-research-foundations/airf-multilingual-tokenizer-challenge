#!/usr/bin/env python3
"""Download one file from the private hidden-evaluation dataset.

Kept separate from the workflow so the auth path is testable. Never prints the
text it downloads, only row counts, so hidden examples cannot leak into CI
logs. Checks the token before reaching for the file, because a bare 401 from
the download does not say whether the token is invalid, belongs to the wrong
account, or simply lacks access to this repository.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.constants import LANGUAGES


def preflight(repo_id: str) -> None:
    """Fail with an actionable message rather than a raw 401."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN is empty. Set it as a repository secret.")
    if token != token.strip():
        raise SystemExit(
            "HF_TOKEN has leading or trailing whitespace, which invalidates it. "
            "Re-set the secret with: printf %s \"$TOKEN\" | gh secret set HF_TOKEN"
        )

    api = HfApi(token=token)
    try:
        identity = api.whoami()
    except HfHubHTTPError as error:
        raise SystemExit(
            f"HF_TOKEN was rejected by Hugging Face ({error.response.status_code}). "
            "It is expired, revoked, or malformed. Create a fresh read token."
        ) from error

    owner = repo_id.split("/")[0]
    accounts = {identity.get("name")} | {
        org["name"] for org in identity.get("orgs", [])
    }
    print(f"token belongs to: {identity.get('name')}")
    if owner not in accounts:
        raise SystemExit(
            f"The token authenticates as {identity.get('name')} but the dataset is "
            f"owned by {owner}. Use a token from the {owner} account, or grant that "
            "account access."
        )

    try:
        api.dataset_info(repo_id)
    except RepositoryNotFoundError as error:
        raise SystemExit(
            f"{repo_id} is not visible to this token. If it is a fine-grained token, "
            "it needs explicit read access to that repository; a classic read token "
            "covers everything the account can see."
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a hidden evaluation split")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    preflight(args.repo_id)

    from huggingface_hub import hf_hub_download

    source = hf_hub_download(
        args.repo_id, args.file, repo_type="dataset",
        token=os.environ["HF_TOKEN"],
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(Path(source).read_bytes())

    with destination.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["language", "text"]:
            raise ValueError("hidden data columns must be exactly: language,text")
        counts = {language: 0 for language in LANGUAGES}
        for row in reader:
            language = (row.get("language") or "").strip()
            if language not in counts:
                raise ValueError(f"unsupported language in hidden data: {language!r}")
            counts[language] += 1

    if len(set(counts.values())) != 1 or not all(counts.values()):
        raise ValueError(f"hidden data must be balanced across languages: {counts}")
    print(f"hidden split: {sum(counts.values()):,} rows, "
          f"{next(iter(counts.values())):,} per language")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
