from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "compose.yaml"
ENV_EXAMPLE = ROOT / ".env.example"
PROMETHEUS = ROOT / "deploy" / "prometheus" / "prometheus.yml"
STACK_DOCS = ROOT / "docs" / "platform" / "local-stack.md"


def document(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def compose() -> dict[str, Any]:
    return document(COMPOSE)


def test_compose_defines_api_dashboard_and_optional_prometheus() -> None:
    services = compose()["services"]

    assert set(services) == {"api", "dashboard", "prometheus"}
    assert services["dashboard"]["profiles"] == ["dashboard"]
    assert services["prometheus"]["profiles"] == ["observability"]
    assert services["dashboard"]["image"] == "semirestore:dashboard"
    assert services["prometheus"]["image"] == "prom/prometheus:v2.55.1"
    assert all(not service.get("image", "").endswith(":latest") for service in services.values())


def test_api_build_is_real_model_cpu_runtime_with_read_only_checkpoint() -> None:
    api = compose()["services"]["api"]

    assert api["build"] == {"context": ".", "dockerfile": "Dockerfile"}
    assert api["image"] == "semirestore:platform"
    assert api["init"] is True
    assert api["stop_grace_period"] == "35s"
    environment = api["environment"]
    assert environment["SEMIRESTORE_DEVICE_PREFERENCE"].endswith(":-cpu}")
    assert environment["SEMIRESTORE_ENABLE_FAKE_MODEL_SERVICE"] == "false"
    assert environment["SEMIRESTORE_CHECKPOINT_PATH"].endswith(
        ":-/models/semirestore_conditioned.pt}"
    )
    checkpoint = api["volumes"][0]
    assert checkpoint["type"] == "bind"
    assert checkpoint["source"].endswith(
        ":-./artifacts/model/semirestore_conditioned.pt}"
    )
    assert checkpoint["target"] == "/models/semirestore_conditioned.pt"
    assert checkpoint["read_only"] is True


def test_health_based_dependencies_and_network_boundaries_are_explicit() -> None:
    config = compose()
    services = config["services"]

    assert "/health/ready" in " ".join(services["api"]["healthcheck"]["test"])
    assert services["dashboard"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert services["prometheus"]["depends_on"]["api"]["condition"] == "service_started"
    assert set(services["api"]["networks"]) == {"backend", "frontend"}
    assert services["dashboard"]["networks"] == ["frontend"]
    assert services["prometheus"]["networks"] == ["backend"]
    assert config["networks"]["backend"]["internal"] is True


def test_dashboard_build_uses_the_frontend_context_and_internal_api_proxy() -> None:
    dashboard = compose()["services"]["dashboard"]

    assert dashboard["build"]["context"] == "./web"
    assert dashboard["build"]["dockerfile"] == "Dockerfile"
    assert dashboard["build"]["args"]["VITE_API_BASE_URL"].endswith(":-/service}")
    assert dashboard["ports"] == ["${SEMIRESTORE_DASHBOARD_PORT:-5173}:8080"]
    assert "127.0.0.1:8080" in " ".join(dashboard["healthcheck"]["test"])
    assert "environment" not in dashboard


def test_services_have_resource_conscious_defaults() -> None:
    services = compose()["services"]

    for name in ("api", "dashboard", "prometheus"):
        assert services[name]["cpus"].startswith("${SEMIRESTORE_")
        assert services[name]["mem_limit"].startswith("${SEMIRESTORE_")
    assert services["api"]["environment"][
        "SEMIRESTORE_INFERENCE_CONCURRENCY_LIMIT"
    ].endswith(":-1}")
    assert services["dashboard"]["read_only"] is True


def test_prometheus_scrapes_only_the_internal_api_metrics_endpoint() -> None:
    config = document(PROMETHEUS)
    jobs = config["scrape_configs"]

    assert len(jobs) == 1
    assert jobs[0]["job_name"] == "semirestore-api"
    assert jobs[0]["metrics_path"] == "/metrics"
    assert jobs[0]["static_configs"] == [{"targets": ["api:8000"]}]
    prometheus_volume = compose()["services"]["prometheus"]["volumes"][0]
    assert prometheus_volume["source"] == "./deploy/prometheus/prometheus.yml"
    assert prometheus_volume["read_only"] is True


def test_example_environment_is_safe_and_contains_no_secrets_or_absolute_paths() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    values = [
        line.split("=", 1)[1]
        for line in text.splitlines()
        if line and not line.startswith("#")
    ]

    assert "SEMIRESTORE_CHECKPOINT_HOST_PATH=" in text
    assert "SEMIRESTORE_DEVICE_PREFERENCE=cpu" in text
    assert "SEMIRESTORE_INFERENCE_CONCURRENCY_LIMIT=1" in text
    assert "SEMIRESTORE_DASHBOARD_API_BASE_URL=/service" in text
    assert not re.search(r"(?i)(password|secret|credential|token|api[_-]?key)=", text)
    assert all(not re.match(r"^(?:[A-Za-z]:[\\/]|/home/|/Users/)", value) for value in values)


def test_compose_contains_no_secret_checkpoint_or_absolute_host_material() -> None:
    text = COMPOSE.read_text(encoding="utf-8")

    assert "best.pt" not in text
    assert "SEMIRESTORE_ENABLE_FAKE_MODEL_SERVICE: \"false\"" in text
    assert not re.search(r"(?i)(password|credential|access[_-]?token|api[_-]?key)", text)
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text)
    assert "/home/" not in text
    assert "/Users/" not in text


def test_local_stack_docs_separate_cpu_and_unimplemented_cuda_paths() -> None:
    text = STACK_DOCS.read_text(encoding="utf-8")

    assert "## CPU startup" in text
    assert "docker compose up --build api" in text
    assert "--profile dashboard" in text
    assert "--profile observability" in text
    assert "## Optional CUDA deployment" in text
    assert "committed image is CPU-only" in text
    assert "No CUDA build or Compose run was performed" in " ".join(text.split())
