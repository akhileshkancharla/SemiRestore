FROM python:3.13-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m pip wheel --no-deps --wheel-dir /wheels .


FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    SEMIRESTORE_ENVIRONMENT=production \
    SEMIRESTORE_HOST=0.0.0.0 \
    SEMIRESTORE_PORT=8000 \
    SEMIRESTORE_DEVICE_PREFERENCE=cpu \
    SEMIRESTORE_MODEL_CONFIG_PATH=/opt/semirestore/model/resolved_conditioned.yaml \
    SEMIRESTORE_MODEL_METADATA_PATH=/opt/semirestore/model/checksums.json \
    SEMIRESTORE_CHECKPOINT_PATH=/models/semirestore_conditioned.pt \
    SEMIRESTORE_ENABLE_FAKE_MODEL_SERVICE=false

WORKDIR /app

RUN groupadd --gid 10001 semirestore \
    && useradd --uid 10001 --gid 10001 --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin semirestore \
    && mkdir -p /opt/semirestore/model /models \
    && chown 10001:10001 /opt/semirestore/model /models

COPY --from=builder /wheels/ /wheels/
COPY --chown=10001:10001 configs/model/resolved_conditioned.yaml \
    /opt/semirestore/model/resolved_conditioned.yaml
COPY --chown=10001:10001 artifacts/model/checksums.json \
    /opt/semirestore/model/checksums.json

RUN set -eu; \
    set -- /wheels/semirestore-*.whl; \
    [ "$#" -eq 1 ] && [ -e "$1" ] \
        || { echo "expected exactly one SemiRestore wheel in /wheels" >&2; exit 1; }; \
    [ -f "$1" ] \
        || { echo "resolved SemiRestore wheel is not a regular file" >&2; exit 1; }; \
    torch_requirement="$(python -c \
        'import email, re, sys, zipfile; archive = zipfile.ZipFile(sys.argv[1]); paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]; assert len(paths) == 1, "expected one metadata file"; metadata = email.message_from_bytes(archive.read(paths[0])); requirements = [value for value in metadata.get_all("Requires-Dist", []) if re.match(r"^torch(?:\s|[<>=!~;(]|$)", value, re.IGNORECASE)]; assert len(requirements) == 1, "expected one torch requirement"; print(requirements[0])' \
        "$1")"; \
    python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "$torch_requirement"; \
    python -m pip install "${1}[platform]"; \
    rm -rf /wheels

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2); raise SystemExit(0 if 200 <= response.status < 300 else 1)"]

CMD ["python", "-m", "uvicorn", "semirestore.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "30"]
