from tokenizers import Tokenizer, models

from competition.constants import LANGUAGES
from competition.data import Example
from competition.metrics import count_characters, measure_fertility, score_tokenizer


def word_tokenizer():
    return Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))


def test_character_count_excludes_all_unicode_whitespace():
    assert count_characters(" a b\tአ\n") == 3


def test_macro_score_is_one_against_itself():
    tokenizer = word_tokenizer()
    examples = [Example(language, "hello") for language in LANGUAGES]
    fertility, tokens, characters = measure_fertility(tokenizer, examples)
    result = score_tokenizer(tokenizer, examples, fertility, benchmark_repeats=1)
    assert result.score == 1.0
    assert set(result.normalized) == set(LANGUAGES)
    assert all(value == 1.0 for value in result.normalized.values())
    assert all(tokens[language] == 1 for language in LANGUAGES)
    assert all(characters[language] == 5 for language in LANGUAGES)

