import json
import shutil
from pathlib import Path

from competition.leaderboard import build_leaderboard, write_leaderboard, write_site_json

ROOT = Path(__file__).resolve().parents[1]


def test_submission_to_all_leaderboard_formats(tmp_path):
    submissions = tmp_path / "submissions"
    baseline = submissions / "baseline"
    challenger = submissions / "test-team"
    shutil.copytree(ROOT / "submissions/baseline", baseline)
    shutil.copytree(ROOT / "submissions/baseline", challenger)
    (challenger / "metadata.yml").write_text(
        "team: Test Team\nmembers:\n  - Test Person\napproach: Contract fixture\n",
        encoding="utf-8",
    )
    draft = submissions / "broken-draft"
    draft.mkdir()
    (draft / "metadata.yml").write_text(
        "team: Broken Draft\nmembers:\n  - Test Person\nfinal: false\n",
        encoding="utf-8",
    )
    (draft / "tokenizer.json").write_text("not valid JSON", encoding="utf-8")

    rows, failures = build_leaderboard(
        submissions,
        ROOT / "tests/fixtures/demo_public_test.csv",
        ROOT / "tests/fixtures/baseline_fertility.json",
        benchmark_repeats=1,
    )
    assert len(failures) == 1
    assert failures[0]["slug"] == "broken-draft"
    assert [row["slug"] for row in rows] == ["test-team", "baseline"]
    assert rows[0]["rank"] == 1
    assert rows[0]["score"] == 1.0
    assert rows[1]["rank"] == "—"

    final_rows, final_failures = build_leaderboard(
        submissions,
        ROOT / "tests/fixtures/demo_public_test.csv",
        ROOT / "tests/fixtures/baseline_fertility.json",
        benchmark_repeats=1,
        final_only=True,
    )
    assert not final_failures
    assert [row["slug"] for row in final_rows] == ["baseline"]

    csv_path = tmp_path / "leaderboard.csv"
    markdown_path = tmp_path / "LEADERBOARD.md"
    json_path = tmp_path / "docs/data/leaderboard.json"
    write_leaderboard(rows, csv_path, markdown_path)
    write_site_json(rows, json_path)
    assert "Test Team" in csv_path.read_text(encoding="utf-8")
    assert "| 1 | Test Team | 1.0000 |" in markdown_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["entries"][0]["team"] == "Test Team"
