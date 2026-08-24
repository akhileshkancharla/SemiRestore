from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts.model import install_local_checkpoint as installer


def _write_metadata(
    path: Path,
    *,
    source: Path,
    destination: Path,
    payload: bytes,
    size_bytes: int | None = None,
    sha256: str | None = None,
) -> Path:
    metadata = {
        "schema_version": 1,
        "checkpoints": {
            "semirestore_conditioned": {
                "source_artifact_path": str(source),
                "runtime_artifact_path": str(destination),
                "size_bytes": len(payload) if size_bytes is None else size_bytes,
                "sha256": hashlib.sha256(payload).hexdigest() if sha256 is None else sha256,
            }
        },
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _fixture_paths(tmp_path: Path, payload: bytes = b"verified synthetic checkpoint") -> tuple:
    source = tmp_path / "source" / "best.pt"
    destination = tmp_path / "runtime" / "semirestore_conditioned.pt"
    metadata = tmp_path / "checksums.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    _write_metadata(
        metadata,
        source=source,
        destination=destination,
        payload=payload,
    )
    return source, destination, metadata, payload


def test_installs_verified_checkpoint_atomically(tmp_path: Path) -> None:
    source, destination, metadata, payload = _fixture_paths(tmp_path)

    result = installer.install_checkpoint(metadata_path=metadata)

    assert result.status == "installed"
    assert result.destination == destination
    assert destination.read_bytes() == payload
    assert source.read_bytes() == payload
    assert list(destination.parent.glob("*.partial")) == []


def test_missing_source_returns_source_error(tmp_path: Path) -> None:
    source, destination, metadata, payload = _fixture_paths(tmp_path)
    source.unlink()

    with pytest.raises(installer.SourceCheckpointError, match="does not exist"):
        installer.install_checkpoint(metadata_path=metadata)

    assert not destination.exists()
    assert payload


def test_source_size_mismatch_is_rejected(tmp_path: Path) -> None:
    source, destination, metadata, payload = _fixture_paths(tmp_path)
    _write_metadata(
        metadata,
        source=source,
        destination=destination,
        payload=payload,
        size_bytes=len(payload) + 1,
    )

    with pytest.raises(installer.CheckpointVerificationError, match="size mismatch"):
        installer.install_checkpoint(metadata_path=metadata)

    assert not destination.exists()


def test_source_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    source, destination, metadata, payload = _fixture_paths(tmp_path)
    _write_metadata(
        metadata,
        source=source,
        destination=destination,
        payload=payload,
        sha256="0" * 64,
    )

    with pytest.raises(installer.CheckpointVerificationError, match="SHA-256 mismatch"):
        installer.install_checkpoint(metadata_path=metadata)

    assert not destination.exists()


def test_existing_verified_destination_is_idempotent(tmp_path: Path) -> None:
    _source, destination, metadata, payload = _fixture_paths(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    original_timestamp = destination.stat().st_mtime_ns

    result = installer.install_checkpoint(metadata_path=metadata)

    assert result.status == "already_installed"
    assert destination.read_bytes() == payload
    assert destination.stat().st_mtime_ns == original_timestamp


def test_conflicting_destination_requires_force(tmp_path: Path) -> None:
    _source, destination, metadata, _payload = _fixture_paths(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"different")

    with pytest.raises(installer.DestinationConflictError, match="--force"):
        installer.install_checkpoint(metadata_path=metadata)

    assert destination.read_bytes() == b"different"


def test_force_atomically_replaces_conflicting_destination(tmp_path: Path) -> None:
    _source, destination, metadata, payload = _fixture_paths(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"different")

    result = installer.install_checkpoint(metadata_path=metadata, force=True)

    assert result.status == "installed"
    assert destination.read_bytes() == payload
    assert list(destination.parent.glob("*.partial")) == []


def test_failed_partial_verification_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, destination, metadata, _payload = _fixture_paths(tmp_path)
    real_file_identity = installer._file_identity

    def mismatched_partial(path: Path) -> installer.FileIdentity:
        identity = real_file_identity(path)
        if path.suffix == ".partial":
            return installer.FileIdentity(identity.size_bytes, "0" * 64)
        return identity

    monkeypatch.setattr(installer, "_file_identity", mismatched_partial)

    with pytest.raises(installer.CheckpointVerificationError, match="Copied checkpoint"):
        installer.install_checkpoint(metadata_path=metadata)

    assert not destination.exists()
    assert list(destination.parent.glob("*.partial")) == []


@pytest.mark.parametrize(
    "document",
    [
        "not-json",
        json.dumps({"schema_version": 1, "checkpoints": {}}),
    ],
)
def test_metadata_loading_failure_is_clear(tmp_path: Path, document: str) -> None:
    metadata = tmp_path / "checksums.json"
    metadata.write_text(document, encoding="utf-8")

    with pytest.raises(installer.MetadataError, match="checkpoint metadata"):
        installer.load_trusted_metadata(metadata)


def test_cli_returns_nonzero_exit_code_without_traceback(tmp_path: Path) -> None:
    source, destination, metadata, payload = _fixture_paths(tmp_path)
    source.unlink()
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = installer.main(
        [],
        metadata_path=metadata,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == installer.ExitCode.SOURCE_ERROR
    assert stdout.getvalue() == ""
    assert stderr.getvalue().startswith("error: Source checkpoint does not exist:")
    assert "Traceback" not in stderr.getvalue()
    assert not destination.exists()
    assert payload
