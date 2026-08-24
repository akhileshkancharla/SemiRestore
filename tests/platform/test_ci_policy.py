from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CI_DOCS = ROOT / "docs" / "platform" / "ci.md"


def workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_ci_has_read_only_permissions_and_superseded_run_cancellation() -> None:
    config = workflow()

    assert config["permissions"] == {"contents": "read"}
    assert config["concurrency"]["cancel-in-progress"] is True
    assert "github.workflow" in config["concurrency"]["group"]
    assert "github.ref" in config["concurrency"]["group"]
    assert set(config["jobs"]) == {"backend-platform", "frontend"}


def test_official_actions_and_runtime_caches_are_pinned_to_stable_majors() -> None:
    config = workflow()
    steps = [step for job in config["jobs"].values() for step in job["steps"]]
    actions = [step["uses"] for step in steps if "uses" in step]

    assert actions.count("actions/checkout@v4") == 2
    assert "actions/setup-python@v5" in actions
    assert "actions/setup-node@v4" in actions
    assert all(re.fullmatch(r"actions/[a-z-]+@v\d+", action) for action in actions)
    python_setup = next(step for step in steps if step.get("uses") == "actions/setup-python@v5")
    node_setup = next(step for step in steps if step.get("uses") == "actions/setup-node@v4")
    assert python_setup["with"]["cache"] == "pip"
    assert node_setup["with"]["cache"] == "npm"


def test_backend_runs_ruff_complete_tests_and_platform_validation_without_checkpoint() -> None:
    backend = workflow()["jobs"]["backend-platform"]
    commands = "\n".join(str(step.get("run", "")) for step in backend["steps"])

    assert 'python -m pip install -e ".[platform,dev]"' in commands
    assert "download.pytorch.org/whl/cpu" in commands
    assert "python -m ruff check ." in commands
    assert "python -m pytest" in commands
    assert "test_container_policy.py" in commands
    assert "test_compose_policy.py" in commands
    assert "test_ci_policy.py" in commands
    assert "docker compose config --quiet" in commands
    assert "checkpoint" not in str(backend.get("env", {})).casefold()


def test_frontend_uses_lockfile_lint_tests_build_and_nonbreaking_audit_threshold() -> None:
    frontend = workflow()["jobs"]["frontend"]
    commands = [step.get("run") for step in frontend["steps"] if "run" in step]

    assert frontend["defaults"]["run"]["working-directory"] == "web"
    assert commands == [
        "npm ci",
        "npm run lint",
        "npm test -- --run",
        "npm run build",
        "npm audit --audit-level=high",
    ]


def test_ci_guards_checkpoints_local_artifacts_and_generated_frontend_outputs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for required in (
        "git ls-files '*.pt' '*.pth' '*.ckpt'",
        "git ls-files 'local/artifacts/**'",
        "git ls-files 'web/dist/**' 'web/coverage/**' 'web/node_modules/**'",
    ):
        assert required in text
    assert "actions/upload-artifact" not in text
    assert not re.search(r"(?i)secrets\.", text)
    assert not re.search(r"(?i)(wget|curl).*(checkpoint|\.pt)", text)


def test_ci_documentation_separates_automated_and_hardware_validation() -> None:
    text = CI_DOCS.read_text(encoding="utf-8")

    assert "## Automated validation" in text
    assert "## Hardware-dependent validation" in text
    assert "does not download a checkpoint" in text
    assert "CUDA" in text
    assert "Docker image build" in text
