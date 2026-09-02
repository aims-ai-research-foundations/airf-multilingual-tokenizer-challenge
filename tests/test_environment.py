import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Editor settings are local organizer convenience and are not published, so
# this check runs for organizers and skips in public CI.
organizer_only = pytest.mark.skipif(
    not (ROOT / ".vscode/settings.json").is_file(),
    reason=".vscode is not part of the public repository",
)


def test_uv_is_the_single_dependency_source():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["uv"]["required-version"] == "0.9.0"
    assert any(requirement.startswith("ipykernel") for requirement in project["dependency-groups"]["dev"])
    assert any(requirement.startswith("datasets") for requirement in project["dependency-groups"]["data"])
    assert (ROOT / "uv.lock").is_file()
    assert not (ROOT / "requirements.txt").exists()


@organizer_only
def test_vscode_discovers_the_uv_environment():
    settings = json.loads((ROOT / ".vscode/settings.json").read_text(encoding="utf-8"))
    assert settings["python.defaultInterpreterPath"] == "${workspaceFolder}/.venv"
    assert "./.venv" in settings["python-envs.workspaceSearchPaths"]
    extensions = json.loads((ROOT / ".vscode/extensions.json").read_text(encoding="utf-8"))
    assert "ms-python.python" in extensions["recommendations"]
    assert "ms-toolsai.jupyter" in extensions["recommendations"]
