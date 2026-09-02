from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tokenizers import Tokenizer, __version__ as tokenizers_version

from .constants import (
    MAX_TOKENIZER_BYTES,
    MAX_VOCAB_SIZE,
    SMOKE_TEXTS,
    SUPPORTED_TOKENIZERS_VERSION,
)


@dataclass
class ValidationReport:
    valid: bool
    path: str
    vocab_size: int | None = None
    file_size_bytes: int | None = None
    tokenizers_version: str = tokenizers_version
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    tokenizer: Tokenizer | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "path": self.path,
            "vocab_size": self.vocab_size,
            "file_size_bytes": self.file_size_bytes,
            "tokenizers_version": self.tokenizers_version,
            "checks": self.checks,
            "errors": self.errors,
        }


def validate_tokenizer(path: str | Path) -> ValidationReport:
    """Apply the same safe, code-free checks used by official evaluation."""
    path = Path(path)
    report = ValidationReport(valid=False, path=str(path))
    if not path.is_file():
        report.errors.append("tokenizer.json was not found")
        return report
    if path.name != "tokenizer.json":
        report.errors.append("submission file must be named tokenizer.json")
        return report

    report.file_size_bytes = path.stat().st_size
    report.checks["file_size"] = report.file_size_bytes <= MAX_TOKENIZER_BYTES
    if not report.checks["file_size"]:
        report.errors.append(f"tokenizer.json exceeds the {MAX_TOKENIZER_BYTES // 1_048_576} MiB limit")
        return report

    try:
        tokenizer = Tokenizer.from_file(str(path))
        report.tokenizer = tokenizer
        report.checks["loads"] = True
    except Exception as exc:  # tokenizers exposes several JSON/model exceptions
        report.checks["loads"] = False
        report.errors.append(f"tokenizer could not be loaded: {exc}")
        return report

    report.vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    report.checks["vocabulary"] = report.vocab_size <= MAX_VOCAB_SIZE
    if not report.checks["vocabulary"]:
        report.errors.append(
            f"vocabulary has {report.vocab_size:,} entries; maximum is {MAX_VOCAB_SIZE:,}"
        )

    try:
        encodings = tokenizer.encode_batch(list(SMOKE_TEXTS.values()), add_special_tokens=False)
        report.checks["encodes_all_languages"] = all(encoding.ids for encoding in encodings)
        if not report.checks["encodes_all_languages"]:
            report.errors.append("one or more required languages produced no tokens")
        decoded = [tokenizer.decode(encoding.ids, skip_special_tokens=False) for encoding in encodings]
        report.checks["decodes"] = all(text.strip() for text in decoded)
        if not report.checks["decodes"]:
            report.errors.append("one or more smoke-test encodings could not be decoded")
    except Exception as exc:
        report.checks["encodes_all_languages"] = False
        report.checks["decodes"] = False
        report.errors.append(f"multilingual encode/decode check failed: {exc}")

    report.checks["compatible_version"] = tokenizers_version == SUPPORTED_TOKENIZERS_VERSION
    if not report.checks["compatible_version"]:
        report.errors.append(
            f"expected tokenizers {SUPPORTED_TOKENIZERS_VERSION}; found {tokenizers_version}"
        )

    report.valid = all(report.checks.values()) and not report.errors
    return report

