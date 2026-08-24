from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"


def test_dashboard_has_repeatable_lint_test_and_build_commands() -> None:
    package = json.loads((WEB / "package.json").read_text(encoding="utf-8"))

    assert package["private"] is True
    assert package["engines"]["node"].startswith(">=24")
    assert set(("lint", "test:run", "build")) <= package["scripts"].keys()
    assert {"react", "react-dom", "react-router-dom"} <= package["dependencies"].keys()


def test_dashboard_container_is_two_stage_and_unprivileged() -> None:
    dockerfile = (WEB / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.count("FROM ") == 2
    assert "node:24-alpine AS builder" in dockerfile
    assert "npm ci" in dockerfile
    assert "npm run build" in dockerfile
    assert "nginxinc/nginx-unprivileged:1.27-alpine" in dockerfile
    assert "EXPOSE 8080" in dockerfile
    assert "USER root" not in dockerfile
    assert ":latest" not in dockerfile


def test_dashboard_runtime_proxies_only_the_service_prefix() -> None:
    nginx = (WEB / "nginx.conf").read_text(encoding="utf-8")

    assert "location /service/" in nginx
    assert "proxy_pass http://api:8000/;" in nginx
    assert "location / {" in nginx
    assert "try_files $uri $uri/ /index.html;" in nginx
    assert "server_tokens off;" in nginx


def test_dashboard_source_uses_typed_client_and_bounded_upload_workflow() -> None:
    client = (WEB / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    types = (WEB / "src" / "api" / "types.ts").read_text(encoding="utf-8")
    inspection = (WEB / "src" / "pages" / "InspectionPage.tsx").read_text(
        encoding="utf-8"
    )
    upload = (WEB / "src" / "components" / "UploadWorkflow.tsx").read_text(
        encoding="utf-8"
    )

    assert "VITE_API_BASE_URL" in client
    assert "ApiRequestError" in client
    assert "interface RestoreResponse" in types
    assert "interface AnalyzeResponse" in types
    assert "UploadWorkflow" in inspection
    assert re.search(r'type="file"', upload)
    assert "SUPPORTED_IMAGE_TYPES" in upload
    assert "AbortController" in upload
    assert "URL.revokeObjectURL" in upload


def test_comparison_uses_exact_png_transport_without_visual_filters() -> None:
    comparison = (WEB / "src" / "components" / "ComparisonWorkspace.tsx").read_text(
        encoding="utf-8"
    )
    transport = (WEB / "src" / "workspace" / "restoredImage.ts").read_text(
        encoding="utf-8"
    )
    styles = (WEB / "src" / "styles.css").read_text(encoding="utf-8")

    assert "restoredPngBlob" in comparison
    assert "Download lossless PNG" in comparison
    assert 'type="range"' in comparison
    assert "URL.revokeObjectURL" in comparison
    assert 'type: "image/png"' in transport
    assert not re.search(r"(?m)^\s*filter\s*:", styles)


def test_assurance_ui_uses_only_returned_advisory_fields() -> None:
    assurance = (WEB / "src" / "components" / "DiagnosticsPanel.tsx").read_text(
        encoding="utf-8"
    )

    assert "result.data.diagnostics" in assurance
    assert "result.data.inference.phase_latency_ms" in assurance
    assert "diagnostics.quality_indicators" in assurance
    assert "diagnostics.spatial" in assurance
    assert "diagnostics.tiles" in assurance
    assert "diagnostics.clipping" in assurance
    assert "not a probability" in assurance
    assert "accuracy" not in assurance.casefold()


def test_compose_keeps_dashboard_optional_and_api_only_independent() -> None:
    config: dict[str, Any] = yaml.safe_load(
        (ROOT / "compose.yaml").read_text(encoding="utf-8")
    )
    services = config["services"]
    dashboard = services["dashboard"]

    assert "profiles" not in services["api"]
    assert dashboard["profiles"] == ["dashboard"]
    assert dashboard["build"]["context"] == "./web"
    assert dashboard["depends_on"]["api"]["condition"] == "service_healthy"
    assert dashboard["read_only"] is True
    assert dashboard["networks"] == ["frontend"]
