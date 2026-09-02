import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_valid_v4_json():
    for name in ("starter.ipynb",):
        payload = json.loads((ROOT / "starter" / name).read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"]
        assert any(cell["cell_type"] == "code" for cell in payload["cells"])


def test_starter_contains_only_participant_notebook():
    assert {path.name for path in (ROOT / "starter").glob("*.ipynb")} == {"starter.ipynb"}


def test_generated_leaderboard_lists_the_baseline_last():
    lines = (ROOT / "LEADERBOARD.md").read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if line.startswith("| ") and "---" not in line]
    assert rows, "LEADERBOARD.md must contain a table"
    assert "(baseline)" in rows[-1]
