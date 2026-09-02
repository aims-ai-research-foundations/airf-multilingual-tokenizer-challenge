#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.constants import LANGUAGES, MAX_VOCAB_SIZE
from competition.data import load_dataset
from competition.metrics import measure_fertility


def train_bpe(data_path: str | Path, vocab_size: int = MAX_VOCAB_SIZE) -> Tokenizer:
    examples = load_dataset(data_path)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        (example.text for example in examples),
        trainer=trainer,
        length=len(examples),
    )
    return tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the official AMTC BPE baseline")
    parser.add_argument("--train-data", default="tests/fixtures/train.csv")
    parser.add_argument("--evaluation-data", default="tests/fixtures/demo_public_test.csv")
    parser.add_argument("--output", default="submissions/baseline/tokenizer.json")
    parser.add_argument("--stats", default="tests/fixtures/baseline_fertility.json")
    parser.add_argument("--vocab-size", type=int, default=MAX_VOCAB_SIZE)
    args = parser.parse_args()
    if not 256 <= args.vocab_size <= MAX_VOCAB_SIZE:
        parser.error(f"--vocab-size must be between 256 and {MAX_VOCAB_SIZE}")

    tokenizer = train_bpe(args.train_data, args.vocab_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output), pretty=True)

    examples = load_dataset(args.evaluation_data)
    fertility, token_counts, character_counts = measure_fertility(tokenizer, examples)
    stats = {
        "metric_version": "1.0",
        "character_definition": "Unicode code points excluding whitespace",
        "evaluation_data": Path(args.evaluation_data).name,
        "fertility": {language: fertility[language] for language in LANGUAGES},
        "token_counts": token_counts,
        "character_counts": character_counts,
    }
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {tokenizer.get_vocab_size():,}-token baseline to {output}")
    print(f"Saved baseline fertility to {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
