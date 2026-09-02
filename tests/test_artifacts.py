import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SECTIONS = {"overview", "challenge", "metric", "data", "starter", "submit", "leaderboard", "rules", "faq"}


class SiteParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.section_ids = set()
        self.tab_targets = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "section" and values.get("id"):
            self.section_ids.add(values["id"])
        if tag == "a" and "nav-tab" in values.get("class", "").split():
            self.tab_targets.add(values.get("href", "").removeprefix("#"))


def test_site_tabs_match_scroll_sections():
    parser = SiteParser()
    parser.feed((ROOT / "docs/index.html").read_text(encoding="utf-8"))
    assert parser.section_ids == EXPECTED_SECTIONS
    assert parser.tab_targets == EXPECTED_SECTIONS
    javascript = (ROOT / "docs/app.js").read_text(encoding="utf-8")
    assert "syncNavigation" in javascript
    assert "aria-current" in javascript


def test_notebooks_are_valid_v4_json():
    for name in ("starter.ipynb",):
        payload = json.loads((ROOT / "starter" / name).read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"]
        assert any(cell["cell_type"] == "code" for cell in payload["cells"])


def test_starter_contains_only_participant_notebook():
    assert {path.name for path in (ROOT / "starter").glob("*.ipynb")} == {"starter.ipynb"}


def test_generated_site_leaderboard_matches_csv():
    payload = json.loads((ROOT / "docs/data/leaderboard.json").read_text(encoding="utf-8"))
    assert payload["entries"]
    assert payload["entries"][-1]["status"] == "baseline"
