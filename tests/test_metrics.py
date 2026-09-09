from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from competition.constants import CONTEXT_FERTILITY_RATIO, SCORED_LANGUAGES
from competition.data import Example, load_dataset
from competition.metrics import (
    competition_score,
    count_words,
    guardrail_breaches,
    measure_fertility,
    round_trip_failures,
    score_tokenizer,
    unknown_token_id,
)

ROOT = Path(__file__).resolve().parents[1]
CHARACTER_LEVEL = ROOT / "starter/baselines/character-level/tokenizer.json"
WORD_LEVEL = ROOT / "starter/baselines/word-level/tokenizer.json"


def word_tokenizer():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def test_word_count_uses_whitespace_separation():
    assert count_words("one two  three") == 3
    assert count_words("  spaced \n out ") == 2
    assert count_words("Ẹ̀kọ́ ṣe pàtàkì") == 3


def test_fertility_and_unknown_rate_are_per_word():
    tokenizer = word_tokenizer()
    examples = [Example("ha", "hello hello"), Example("ha", "hello unseen")]
    fertility, unknown_rate, tokens, words = measure_fertility(tokenizer, examples)
    assert words["ha"] == 4
    assert tokens["ha"] == 4
    assert fertility["ha"] == pytest.approx(1.0)
    assert unknown_rate["ha"] == pytest.approx(0.25)


def test_score_averages_only_the_four_target_languages():
    fertility = {"en": 9.0, "fr": 9.0, "ha": 1.0, "sw": 2.0, "yo": 3.0, "am": 4.0}
    assert competition_score(fertility) == pytest.approx(2.5)
    assert set(SCORED_LANGUAGES) == {"ha", "sw", "yo", "am"}


def test_unknown_tokens_add_a_hundred_times_their_rate():
    fertility = dict.fromkeys(("en", "fr", "ha", "sw", "yo", "am"), 2.0)
    assert competition_score(fertility, {}) == pytest.approx(2.0)

    one_percent = dict.fromkeys(SCORED_LANGUAGES, 0.01)
    assert competition_score(fertility, one_percent) == pytest.approx(3.0)

    ten_percent = dict.fromkeys(SCORED_LANGUAGES, 0.10)
    assert competition_score(fertility, ten_percent) == pytest.approx(12.0)


def test_score_rejects_a_missing_target_language():
    with pytest.raises(ValueError, match="missing scored languages"):
        competition_score({"ha": 1.0, "sw": 1.0, "yo": 1.0})


def test_guardrail_is_relative_to_the_scored_average():
    balanced = {"en": 2.0, "fr": 2.0, "ha": 2.0, "sw": 2.0, "yo": 2.0, "am": 2.0}
    assert guardrail_breaches(balanced) == []

    limit = 2.0 * CONTEXT_FERTILITY_RATIO
    sacrificed = {**balanced, "fr": limit + 0.01}
    assert guardrail_breaches(sacrificed) == ["fr"]


def test_unknown_token_id_is_discovered_from_the_model():
    assert unknown_token_id(word_tokenizer()) == 0


def test_word_level_is_penalised_and_character_level_is_not():
    examples = load_dataset(ROOT / "tests/fixtures/validation.csv")

    word_level = Tokenizer.from_file(str(WORD_LEVEL))
    word_result = score_tokenizer(word_level, examples, benchmark_repeats=1)
    assert word_result.unknown_tokens > 0
    # Low raw fertility, but the unknown-token penalty dominates the score.
    assert word_result.fertility["ha"] < 2.0
    assert word_result.score > word_result.fertility["ha"] * 5

    character_level = Tokenizer.from_file(str(CHARACTER_LEVEL))
    character_result = score_tokenizer(character_level, examples, benchmark_repeats=1)
    assert character_result.unknown_tokens == 0
    assert round_trip_failures(character_level, examples) == []
    assert character_result.score == pytest.approx(
        competition_score(character_result.fertility, character_result.unknown_rate)
    )
