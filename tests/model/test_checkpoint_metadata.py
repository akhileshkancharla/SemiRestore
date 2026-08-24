from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

METADATA_PATH = Path("artifacts/model/checksums.json")
EXPECTED_SOURCE = Path("local/artifacts/naf_sr_conditioned_d037473/best.pt")
EXPECTED_RUNTIME = Path("artifacts/model/semirestore_conditioned.pt")
EXPECTED_SHA256 = "273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28"
EXPECTED_SIZE = 36_565_383


def _checkpoint_metadata() -> dict[str, object]:
    document = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    return document["checkpoints"]["semirestore_conditioned"]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_checkpoint_metadata_contract() -> None:
    metadata = _checkpoint_metadata()

    assert metadata == {
        "model_name": "naf_sr",
        "model_version": "conditioned-d037473",
        "architecture": "statistics-conditioned NAF-SR",
        "scale": 2,
        "expected_parameter_count": 9_111_684,
        "source_artifact_path": EXPECTED_SOURCE.as_posix(),
        "runtime_artifact_path": EXPECTED_RUNTIME.as_posix(),
        "sha256": EXPECTED_SHA256,
        "size_bytes": EXPECTED_SIZE,
        "training_revision": "d037473ddf4a3cd20eb3fef933991cd66749f4f2",
        "checkpoint_role": "deployable best-validation weights",
        "verification_status": "verified against immutable local source",
        "binary_storage": "ignored runtime artifact; excluded from Git",
    }


def test_local_source_matches_registered_identity_when_available() -> None:
    if not EXPECTED_SOURCE.is_file():
        pytest.skip("immutable local checkpoint source is unavailable")

    assert EXPECTED_SOURCE.stat().st_size == EXPECTED_SIZE
    assert _sha256(EXPECTED_SOURCE) == EXPECTED_SHA256
