import csv

from competition.constants import LANGUAGES
from scripts.prepare_dataset import prepare


def test_prepare_dataset_is_balanced_and_separates_splits(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    for language in LANGUAGES:
        (source / f"{language}.txt").write_text(
            "\n".join(f"{language} unique sentence {index}" for index in range(4)) + "\n",
            encoding="utf-8",
        )
    manifest = prepare(
        source,
        output,
        {"train": 1, "validation": 1, "public_test": 1, "private_test": 1},
        seed=7,
    )
    assert set(manifest) == {"train", "validation", "public_test", "private_test"}
    seen = set()
    for split in manifest:
        with (output / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(LANGUAGES)
        assert {row["language"] for row in rows} == set(LANGUAGES)
        texts = {row["text"] for row in rows}
        assert not seen.intersection(texts)
        seen.update(texts)

