import pytest

from competition.pr import submission_slug_from_changes


def test_one_team_submission_is_accepted():
    assert submission_slug_from_changes([
        "submissions/token-lab/tokenizer.json",
        "submissions/token-lab/metadata.yml",
    ]) == "token-lab"


@pytest.mark.parametrize("files", [
    ["README.md", "submissions/token-lab/tokenizer.json"],
    ["submissions/one/tokenizer.json", "submissions/two/tokenizer.json"],
    ["submissions/baseline/tokenizer.json"],
    ["submissions/token-lab/run.py"],
])
def test_unscoped_submission_changes_are_rejected(files):
    with pytest.raises(ValueError):
        submission_slug_from_changes(files)

