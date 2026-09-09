#!/usr/bin/env python3
"""Build the two reference baselines: word level and character level.

They bracket the design problem. A word-level tokenizer needs very few tokens
per word but cannot represent words it has never seen, so it fails the coverage
requirement. A character-level tokenizer represents everything but needs many
tokens per word. A good submission sits between them.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.constants import MAX_VOCAB_SIZE
from competition.data import load_dataset


def train_word_level(texts: list[str], vocab_size: int = MAX_VOCAB_SIZE) -> Tokenizer:
    """Give each frequent word a single token, with [UNK] for everything else."""
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size, special_tokens=["[UNK]"], show_progress=False
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    return tokenizer


def train_character_level(texts: list[str], min_count: int = 20) -> Tokenizer:
    """Give each common character a single token, falling back to raw bytes.

    Byte fallback is what makes this baseline satisfy the coverage requirement:
    any character missing from the vocabulary is still representable.
    """
    counts = Counter(character for text in texts for character in text)
    alphabet = [character for character, seen in counts.most_common() if seen >= min_count]

    vocab = {"[UNK]": 0}
    for character in alphabet:
        vocab.setdefault(character, len(vocab))
    for value in range(256):
        vocab.setdefault(f"<0x{value:02X}>", len(vocab))

    tokenizer = Tokenizer(
        models.BPE(vocab=vocab, merges=[], unk_token="[UNK]", byte_fallback=True)
    )
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.decoder = decoders.ByteFallback()
    return tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the AMTC reference baselines")
    parser.add_argument("--train-data", default="tests/fixtures/train.csv")
    parser.add_argument("--output-dir", default="starter/baselines")
    parser.add_argument("--vocab-size", type=int, default=MAX_VOCAB_SIZE)
    args = parser.parse_args()

    texts = [example.text for example in load_dataset(args.train_data)]
    output = Path(args.output_dir)

    output.mkdir(parents=True, exist_ok=True)
    for name, tokenizer in (
        ("word-level", train_word_level(texts, args.vocab_size)),
        ("character-level", train_character_level(texts)),
    ):
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "tokenizer.json"
        tokenizer.save(str(path), pretty=True)
        print(f"{name}: {tokenizer.get_vocab_size(True):,} tokens -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
