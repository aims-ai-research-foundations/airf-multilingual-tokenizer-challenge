import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from scripts.build_hf_dataset import (
    CARD_TEMPLATE,
    MAX_CHARACTERS,
    MIN_CHARACTERS,
    SOURCE_REVISION,
    TARGET_CHARACTERS,
    TARGETS,
    choose_split,
    content_key,
    normalize_text,
    segment_article,
)


# The card template is organizer-only material kept outside the published
# repository, so these checks run for organizers and skip in public CI.
organizer_only = pytest.mark.skipif(
    not CARD_TEMPLATE.is_file(),
    reason="organizer-private/CARD_TEMPLATE.md is not part of the public repository",
)


@organizer_only
def test_dataset_card_declares_only_public_splits():
    card = CARD_TEMPLATE.read_text(encoding="utf-8")
    assert "path: data/train-*.parquet" in card
    assert "path: data/validation-*.parquet" in card
    assert "public_test-*.parquet" not in card
    assert "private_test-*.parquet" not in card
    assert SOURCE_REVISION in card


def test_segmentation_preserves_unicode_and_obeys_bounds():
    sentence = "Ẹ̀kọ́ Yorùbá àti Cafe\u0301 ṣe pàtàkì gan-an fun gbogbo wa. "
    passages = segment_article(sentence * 30)
    assert passages
    assert all(MIN_CHARACTERS <= len(text) <= MAX_CHARACTERS for text in passages)
    assert sum(len(text) for text in passages) / len(passages) <= TARGET_CHARACTERS * 1.5
    assert all(text == unicodedata.normalize("NFC", text) for text in passages)
    assert "Café" in " ".join(passages)
    assert "Cafe\u0301" not in " ".join(passages)


def test_normalized_content_hash_deduplicates_case_and_spacing():
    assert content_key("  CAFÉ\n") == content_key("café")
    assert normalize_text(" a\x00  b\n c ") == "a b c"


def test_split_choice_never_selects_a_completed_split():
    counts = Counter({"train": TARGETS["train"]})
    for source_id in ("1", "2", "3", "4"):
        assert choose_split("ha", source_id, counts) != "train"


@organizer_only
def test_release_template_is_kept_outside_the_public_repository():
    root = Path(__file__).resolve().parents[1]
    assert CARD_TEMPLATE == root / "organizer-private" / "CARD_TEMPLATE.md"
    assert CARD_TEMPLATE.is_file()
