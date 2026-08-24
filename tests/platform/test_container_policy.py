from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
PYPROJECT = ROOT / "pyproject.toml"


def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def logical_instructions(text: str) -> list[str]:
    instructions: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        current = f"{current} {line}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        instructions.append(current)
        current = ""
    if current:
        instructions.append(current)
    return instructions


def ignore_patterns() -> set[str]:
    return {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def runtime_instructions() -> list[str]:
    instructions = logical_instructions(dockerfile_text())
    final_from = max(
        index for index, instruction in enumerate(instructions) if instruction.startswith("FROM ")
    )
    return instructions[final_from:]


def test_container_policy_files_exist() -> None:
    assert DOCKERFILE.is_file()
    assert DOCKERIGNORE.is_file()


def test_dockerfile_uses_explicit_supported_python_slim_stages() -> None:
    from_instructions = [
        instruction
        for instruction in logical_instructions(dockerfile_text())
        if instruction.startswith("FROM ")
    ]

    assert len(from_instructions) == 2
    assert all(
        re.match(r"FROM python:3\.13-slim-[a-z]+(?: AS [a-z]+)?$", instruction)
        for instruction in from_instructions
    )
    assert from_instructions[0].endswith(" AS builder")
    assert from_instructions[1].endswith(" AS runtime")


def test_runtime_has_safe_python_pip_and_cpu_environment() -> None:
    runtime = "\n".join(runtime_instructions())

    assert "PYTHONDONTWRITEBYTECODE=1" in runtime
    assert "PYTHONUNBUFFERED=1" in runtime
    assert "PIP_DISABLE_PIP_VERSION_CHECK=1" in runtime
    assert "PIP_NO_CACHE_DIR=1" in runtime
    assert "SEMIRESTORE_ENVIRONMENT=production" in runtime
    assert "SEMIRESTORE_DEVICE_PREFERENCE=cpu" in runtime
    assert "SEMIRESTORE_ENABLE_FAKE_MODEL_SERVICE=false" in runtime
    assert "WORKDIR /app" in runtime
    assert "EXPOSE 8000" in runtime


def test_project_wheel_and_platform_dependencies_are_installed() -> None:
    text = dockerfile_text()
    normalized = " ".join(text.split())

    assert "COPY pyproject.toml README.md ./" in normalized
    assert "COPY src/ ./src/" in normalized
    assert "pip wheel --no-deps --wheel-dir /wheels ." in normalized
    assert "COPY --from=builder /wheels/ /wheels/" in normalized
    assert "https://download.pytorch.org/whl/cpu" in normalized

    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    platform_dependencies = project["project"]["optional-dependencies"]["platform"]
    assert any(dependency.startswith("uvicorn>=0.50,<1") for dependency in platform_dependencies)
    assert not any("gunicorn" in dependency.lower() for dependency in platform_dependencies)


def test_runtime_resolves_exactly_one_version_independent_project_wheel() -> None:
    text = dockerfile_text()
    normalized = " ".join(text.split())

    assert re.search(r"semirestore-\d+\.\d+", text, re.IGNORECASE) is None
    assert "set -- /wheels/semirestore-*.whl" in normalized
    assert '[ "$#" -eq 1 ]' in normalized
    assert '[ -e "$1" ]' in normalized
    assert '[ -f "$1" ]' in normalized
    assert '"${1}[platform]"' in normalized
    assert "expected exactly one SemiRestore wheel" in normalized
    assert "not a regular file" in normalized


def test_runtime_installs_only_the_validated_project_wheel() -> None:
    runtime = " ".join(runtime_instructions())

    assert not re.search(r"pip install [^;]*(?<!semirestore-)\*\.whl", runtime)
    assert "pip install /wheels/*.whl" not in runtime
    assert "pip install /wheels/semirestore-*.whl" not in runtime
    assert 'pip install "${1}[platform]"' in runtime
    assert "torch_requirement" in runtime
    assert "Requires-Dist" in runtime
    assert '"torch>=2.6,<3"' not in runtime


def test_runtime_is_non_root_before_exec_form_single_worker_startup() -> None:
    runtime = runtime_instructions()
    user_indexes = [
        index for index, instruction in enumerate(runtime) if instruction.startswith("USER ")
    ]
    command_indexes = [
        index for index, instruction in enumerate(runtime) if instruction.startswith("CMD ")
    ]

    assert user_indexes
    assert command_indexes
    assert runtime[user_indexes[-1]] == "USER 10001:10001"
    assert user_indexes[-1] < command_indexes[-1]
    command = runtime[command_indexes[-1]]
    assert command.startswith("CMD [")
    assert '"semirestore.api:create_app"' in command
    assert '"--factory"' in command
    assert re.search(r'"--workers"\s*,\s*"1"', command)
    assert "--reload" not in command
    assert "gunicorn" not in command.lower()


def test_healthcheck_uses_stdlib_liveness_with_bounded_policy() -> None:
    healthchecks = [
        instruction
        for instruction in runtime_instructions()
        if instruction.startswith("HEALTHCHECK ")
    ]

    assert len(healthchecks) == 1
    healthcheck = healthchecks[0]
    assert "/health/live" in healthcheck
    assert "/health/ready" not in healthcheck
    assert "urllib.request" in healthcheck
    assert "127.0.0.1:8000" in healthcheck
    for policy in ("--interval=30s", "--timeout=3s", "--start-period=20s", "--retries=3"):
        assert policy in healthcheck


def test_dockerfile_never_copies_forbidden_runtime_material() -> None:
    instructions = logical_instructions(dockerfile_text())
    copies = [
        instruction.lower()
        for instruction in instructions
        if instruction.startswith("COPY ")
    ]

    assert not any(re.match(r"copy\s+(?:--\S+\s+)*\.\s+", instruction) for instruction in copies)
    for forbidden in (
        "local/",
        "local/artifacts",
        "artifacts/",
        ".env",
        ".pt",
        ".pth",
        ".ckpt",
        "uploads/",
        "outputs/",
        "runs/",
    ):
        assert all(forbidden not in instruction for instruction in copies)


def test_dockerfile_has_no_secret_arguments_network_tools_or_gpu_runtime() -> None:
    instructions = logical_instructions(dockerfile_text())
    environment = [
        instruction.lower()
        for instruction in instructions
        if instruction.startswith(("ENV ", "ARG "))
    ]
    sensitive_names = re.compile(r"(password|secret|credential|access[_-]?token|api[_-]?key)")

    assert all(sensitive_names.search(instruction) is None for instruction in environment)
    text = dockerfile_text().lower()
    assert "curl" not in text
    assert "wget" not in text
    assert "apt-get" not in text
    assert "cuda" not in text
    assert "nvidia" not in text


def test_dockerignore_blocks_runtime_artifacts_and_local_state() -> None:
    patterns = ignore_patterns()

    for required in (
        ".git/",
        ".github/",
        ".venv/",
        "local/",
        "local/brand/",
        "local/artifacts/",
        "artifacts/",
        "outputs/",
        "uploads/",
        "runs/",
        "Microsoft/",
        ".env",
        ".pytest_cache/",
        ".ruff_cache/",
        ".coverage",
        "*.log",
        "*.png",
        "*.tmp",
        "*credentials*",
        "*secret*",
    ):
        assert required in patterns
    for extension in ("*.pt", "*.pth", "*.ckpt"):
        assert extension in patterns


def test_dockerignore_keeps_required_build_inputs() -> None:
    patterns = ignore_patterns()

    assert "src/" not in patterns
    assert "pyproject.toml" not in patterns
    assert "README.md" not in patterns
    assert "!src/" not in patterns
