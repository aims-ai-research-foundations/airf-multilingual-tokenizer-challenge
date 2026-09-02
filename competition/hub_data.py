"""Canonical public competition dataset location and lazy loader."""
from __future__ import annotations

PUBLIC_DATASET_ID = "Similoluwa/african-multilingual-tokenizer-challenge"
PUBLIC_DATASET_REVISION = "v1.0.0"


def load_public_split(split: str, *, streaming: bool = False, revision: str = PUBLIC_DATASET_REVISION):
    if split not in {"train", "validation"}:
        raise ValueError("Only train and validation are public")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Run `uv sync --dev --group data` to install dataset support") from exc
    return load_dataset(
        PUBLIC_DATASET_ID,
        split=split,
        revision=revision,
        streaming=streaming,
    )

