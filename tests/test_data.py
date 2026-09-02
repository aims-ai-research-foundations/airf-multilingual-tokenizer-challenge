from pathlib import Path

import pytest

from competition.constants import LANGUAGES
from competition.data import language_counts, load_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative",
    [
        "tests/fixtures/train.csv",
        "tests/fixtures/validation.csv",
        "tests/fixtures/demo_public_test.csv",
    ],
)
def test_bundled_datasets_are_balanced_and_complete(relative):
    examples = load_dataset(ROOT / relative)
    counts = language_counts(examples)
    assert set(counts) == set(LANGUAGES)
    assert len(set(counts.values())) == 1
    assert all(count > 0 for count in counts.values())


def test_dataset_requires_exact_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("text,language\nhello,en\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        load_dataset(path)

