from __future__ import annotations

import re
from pathlib import Path

from semirestore.api import create_app
from semirestore.platform import RuntimeSettings

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_DOCS = ROOT / "docs" / "platform"

REQUIRED_DOCUMENTS = {
    "README.md",
    "architecture.md",
    "api-contract.md",
    "deployment.md",
    "environment.md",
    "integration-checklist.md",
    "local-stack.md",
    "runbook.md",
    "smoke-testing.md",
    "troubleshooting.md",
}

PUBLIC_ENDPOINTS = {
    "/health/live",
    "/health/ready",
    "/health/model",
    "/version",
    "/metrics",
    "/api/v1/analyze",
    "/api/v1/restore",
    "/api/v1/restore-and-analyze",
}


def _all_handoff_text() -> str:
    return "\n".join(
        (PLATFORM_DOCS / name).read_text(encoding="utf-8") for name in REQUIRED_DOCUMENTS
    )


def test_required_handoff_documents_exist_and_are_indexed() -> None:
    assert REQUIRED_DOCUMENTS <= {path.name for path in PLATFORM_DOCS.glob("*.md")}
    index = (PLATFORM_DOCS / "README.md").read_text(encoding="utf-8")
    for name in REQUIRED_DOCUMENTS - {"README.md"}:
        assert f"({name}" in index


def test_documented_endpoint_names_match_application_routes() -> None:
    app = create_app(settings=RuntimeSettings())
    candidates = list(app.routes)
    for route in app.routes:
        candidates.extend(getattr(getattr(route, "original_router", None), "routes", ()))
    actual_paths = {
        path for candidate in candidates if (path := getattr(candidate, "path", None))
    }
    assert PUBLIC_ENDPOINTS <= actual_paths

    handoff = _all_handoff_text()
    for endpoint in PUBLIC_ENDPOINTS:
        assert f"`{endpoint}`" in handoff or endpoint in handoff

    documented_api_paths = set(re.findall(r"/api/v1/[a-z-]+", handoff))
    assert documented_api_paths == {
        "/api/v1/analyze",
        "/api/v1/restore",
        "/api/v1/restore-and-analyze",
    }


def test_environment_reference_covers_every_runtime_setting() -> None:
    reference = (PLATFORM_DOCS / "environment.md").read_text(encoding="utf-8")
    expected = {f"SEMIRESTORE_{name.upper()}" for name in RuntimeSettings.model_fields}
    documented = set(re.findall(r"SEMIRESTORE_[A-Z0-9_]+", reference))
    assert expected <= documented


def test_handoff_includes_required_release_safety_guidance() -> None:
    handoff = _all_handoff_text().lower()
    required_phrases = {
        "not ground truth",
        "advisory",
        "not permanently stored",
        "one worker",
        "prometheus",
        "graceful shutdown",
        "rollback",
        "read-only",
        "no benchmark result is claimed",
        "cuda",
        "not performed",
    }
    assert required_phrases <= {phrase for phrase in required_phrases if phrase in handoff}


def test_root_readme_has_setup_test_demo_and_handoff_links() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Python 3.11 or newer" in readme
    assert "scripts/model/install_local_checkpoint.py" in readme
    assert "semirestore.api:create_app --factory" in readme
    assert "scripts/platform/smoke_test.py" in readme
    assert "python.exe -m pytest" in readme
    assert "npm test -- --run" in readme
    assert "docs/platform/runbook.md" in readme
    assert "docs/platform/troubleshooting.md" in readme


def test_handoff_does_not_describe_the_integrated_adapter_as_future_work() -> None:
    text = ((ROOT / "README.md").read_text(encoding="utf-8") + _all_handoff_text()).lower()
    prohibited = {
        "future model adapter",
        "until the real adapter is integrated",
        "final pipeline is not available",
        "image upload and the complete restoration workspace remain",
    }
    assert not {phrase for phrase in prohibited if phrase in text}
