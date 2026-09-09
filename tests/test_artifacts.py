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


def test_reference_baselines_are_published_with_the_starter_kit():
    baselines = ROOT / "starter/baselines"
    assert {path.parent.name for path in baselines.glob("*/tokenizer.json")} == {
        "word-level", "character-level"
    }


def test_generated_leaderboard_has_a_table_header():
    text = (ROOT / "LEADERBOARD.md").read_text(encoding="utf-8")
    assert "Score" in text and "Hausa" in text and "Amharic" in text
