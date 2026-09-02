from pathlib import Path

from tokenizers import Tokenizer, models

from competition.cli import check_main
from competition.constants import MAX_VOCAB_SIZE
from competition.validation import validate_tokenizer


def save_wordlevel(path: Path, size: int):
    vocab = {"[UNK]": 0, **{f"token-{index}": index for index in range(1, size)}}
    Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")).save(str(path))


def test_baseline_passes():
    root = Path(__file__).resolve().parents[1]
    report = validate_tokenizer(root / "submissions/baseline/tokenizer.json")
    assert report.valid, report.errors
    assert report.vocab_size <= MAX_VOCAB_SIZE


def test_oversized_vocabulary_fails(tmp_path):
    path = tmp_path / "tokenizer.json"
    save_wordlevel(path, MAX_VOCAB_SIZE + 1)
    report = validate_tokenizer(path)
    assert not report.valid
    assert not report.checks["vocabulary"]
    assert "maximum" in report.errors[0]


def test_wrong_filename_fails(tmp_path):
    path = tmp_path / "model.json"
    save_wordlevel(path, 2)
    report = validate_tokenizer(path)
    assert not report.valid
    assert "named tokenizer.json" in report.errors[0]


def test_participant_checker_reports_local_benchmark(capsys):
    root = Path(__file__).resolve().parents[1]
    exit_code = check_main([str(root / "submissions/baseline/tokenizer.json")])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Local benchmark (informational)" in output
    assert "READY FOR SUBMISSION" in output
