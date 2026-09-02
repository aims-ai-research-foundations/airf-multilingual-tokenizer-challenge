from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_github_workflows_are_valid_yaml_documents():
    workflow_dir = ROOT / ".github/workflows"
    workflows = list(workflow_dir.glob("*.yml"))
    assert {path.name for path in workflows} == {
        "evaluate.yml",
        "final-evaluation.yml",
        "validate-submission.yml",
    }
    for path in workflows:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        assert payload["name"]
        assert "on" in payload
        assert payload["jobs"]

