from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_METADATA_BYTES = 16 * 1024


@dataclass(frozen=True)
class Metadata:
    team: str
    members: tuple[str, ...]
    affiliation: str = ""
    approach: str = ""
    final: bool = False


def load_metadata(path: str | Path) -> Metadata:
    path = Path(path)
    if not path.is_file():
        raise ValueError("metadata.yml was not found")
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("metadata.yml exceeds the 16 KiB limit")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metadata.yml must contain a mapping")
    team = payload.get("team")
    members = payload.get("members")
    if not isinstance(team, str) or not team.strip() or len(team.strip()) > 80:
        raise ValueError("team must be a non-empty string of at most 80 characters")
    if not isinstance(members, list) or not members or not all(
        isinstance(member, str) and member.strip() for member in members
    ):
        raise ValueError("members must be a non-empty list of names")
    if len(members) > 6:
        raise ValueError("a team may have at most 6 members")
    text_fields = [team, *members]
    if any(any(character in "\r\n" or unicodedata.category(character) == "Cc" for character in value) for value in text_fields):
        raise ValueError("team and member names may not contain newlines or control characters")
    return Metadata(
        team=team.strip(),
        members=tuple(member.strip() for member in members),
        affiliation=str(payload.get("affiliation", "")).strip()[:120],
        approach=str(payload.get("approach", "")).strip()[:240],
        final=bool(payload.get("final", False)),
    )


def validate_submission_directory(path: str | Path) -> tuple[Metadata, Path]:
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"submission directory not found: {path}")
    if not SLUG_PATTERN.fullmatch(path.name):
        raise ValueError("submission folder must be a lowercase kebab-case team slug")
    allowed = {"tokenizer.json", "metadata.yml", "README.md"}
    symlinks = sorted(item.name for item in path.iterdir() if item.is_symlink())
    if symlinks:
        raise ValueError(f"symlinks are not allowed: {', '.join(symlinks)}")
    unexpected = sorted(item.name for item in path.iterdir() if item.name not in allowed)
    if unexpected:
        raise ValueError(f"unexpected submission files: {', '.join(unexpected)}")
    metadata = load_metadata(path / "metadata.yml")
    tokenizer_path = path / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise ValueError("tokenizer.json was not found")
    return metadata, tokenizer_path
