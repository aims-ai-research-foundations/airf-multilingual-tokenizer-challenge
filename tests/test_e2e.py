import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from competition.constants import LANGUAGES
from competition.data import load_dataset

from competition.leaderboard import build_leaderboard, write_leaderboard

ROOT = Path(__file__).resolve().parents[1]


def _fixture_tokenizer():
    """Train a small byte level BPE on the bundled fixture data."""
    texts = [item.text for item in load_dataset(ROOT / "tests/fixtures/train.csv")]
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainer=trainers.BpeTrainer(
        vocab_size=800, min_frequency=1, special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    ), length=len(texts))
    return tokenizer


def test_submission_to_all_leaderboard_formats(tmp_path):
    submissions = tmp_path / "submissions"
    baseline = submissions / "baseline"
    challenger = submissions / "test-team"
    tokenizer = _fixture_tokenizer()
    for directory, team in ((baseline, "Reference"), (challenger, "Test Team")):
        directory.mkdir(parents=True)
        tokenizer.save(str(directory / "tokenizer.json"))
        (directory / "metadata.yml").write_text(
            f"team: {team}\nmembers:\n  - Test Person\n", encoding="utf-8")
    draft = submissions / "broken-draft"
    draft.mkdir()
    (draft / "metadata.yml").write_text(
        "team: Broken Draft\nmembers:\n  - Test Person\n",
        encoding="utf-8",
    )
    (draft / "tokenizer.json").write_text("not valid JSON", encoding="utf-8")

    rows, failures = build_leaderboard(
        submissions,
        ROOT / "tests/fixtures/demo_public_test.csv",
        benchmark_repeats=1,
    )
    assert len(failures) == 1
    assert failures[0]["slug"] == "broken-draft"
    assert {row["slug"] for row in rows} == {"test-team", "baseline"}
    assert rows[0]["rank"] == 1

    csv_path = tmp_path / "leaderboard.csv"
    markdown_path = tmp_path / "LEADERBOARD.md"
    json_path = tmp_path / "leaderboard.json"
    write_leaderboard(rows, csv_path, markdown_path, json_path=json_path)
    assert "Test Team" in csv_path.read_text(encoding="utf-8")
    assert "Test Team" in markdown_path.read_text(encoding="utf-8")

    # The JSON feed is a published contract: the website reads these keys.
    feed = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(feed) == {"generated_at", "metric", "repository", "entries"}
    assert len(feed["entries"]) == len(rows)
    ranked = [entry for entry in feed["entries"] if entry["status"] == "ranked"]
    assert [entry["team"] for entry in ranked] == ["Test Team"]
    entry = ranked[0]
    assert set(entry) == {
        "rank", "team", "slug", "status", "score", "fertility", "unknown_rate",
        "speed_chars_per_second", "vocab_size", "evaluated_at",
        "fertility_by_language",
    }
    assert entry["rank"] == 1
    assert entry["score"] > 0
    assert set(entry["fertility_by_language"]) == set(LANGUAGES)
