"""Verify and atomically install the local SemiRestore deployment checkpoint.

This installer treats checkpoint files as opaque bytes. It never deserializes
PyTorch content and obtains the trusted size and digest from the tracked
``artifacts/model/checksums.json`` manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "model" / "checksums.json"
CHECKPOINT_KEY = "semirestore_conditioned"
COPY_CHUNK_SIZE = 1024 * 1024


class ExitCode(IntEnum):
    """Stable process exit codes for expected installer failures."""

    SUCCESS = 0
    METADATA_ERROR = 3
    SOURCE_ERROR = 4
    VERIFICATION_ERROR = 5
    DESTINATION_CONFLICT = 6
    INSTALLATION_ERROR = 7


class InstallerError(RuntimeError):
    """Base class for safe, expected installer failures."""

    exit_code = ExitCode.INSTALLATION_ERROR


class MetadataError(InstallerError):
    """Raised when trusted checkpoint metadata cannot be loaded or validated."""

    exit_code = ExitCode.METADATA_ERROR


class SourceCheckpointError(InstallerError):
    """Raised when the requested source is absent or not a regular file."""

    exit_code = ExitCode.SOURCE_ERROR


class CheckpointVerificationError(InstallerError):
    """Raised when an opaque checkpoint does not match trusted metadata."""

    exit_code = ExitCode.VERIFICATION_ERROR


class DestinationConflictError(InstallerError):
    """Raised when a different destination exists and force was not requested."""

    exit_code = ExitCode.DESTINATION_CONFLICT


class InstallationError(InstallerError):
    """Raised when a verified checkpoint cannot be copied or installed."""

    exit_code = ExitCode.INSTALLATION_ERROR


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Trusted identity and default paths loaded from the tracked manifest."""

    source_path: Path
    runtime_path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Opaque file size and incremental SHA-256."""

    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class InstallationResult:
    """Outcome of an idempotent or newly completed installation."""

    status: str
    destination: Path
    identity: FileIdentity


def _manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _required_string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise MetadataError(f"Checkpoint metadata field {key!r} must be a non-empty string")
    return value


def load_trusted_metadata(
    path: str | Path = DEFAULT_METADATA_PATH,
) -> CheckpointMetadata:
    """Load and validate the conditioned checkpoint entry from tracked JSON."""

    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise MetadataError(f"Trusted checkpoint metadata is missing: {metadata_path}")
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MetadataError(
            f"Could not read trusted checkpoint metadata: {metadata_path}"
        ) from error

    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise MetadataError("Trusted checkpoint metadata has an unsupported schema")
    checkpoints = document.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise MetadataError("Trusted checkpoint metadata has no checkpoints mapping")
    values = checkpoints.get(CHECKPOINT_KEY)
    if not isinstance(values, Mapping):
        raise MetadataError(f"Trusted checkpoint metadata has no {CHECKPOINT_KEY!r} entry")

    size_bytes = values.get("size_bytes")
    if type(size_bytes) is not int or size_bytes < 1:
        raise MetadataError("Checkpoint metadata field 'size_bytes' must be a positive integer")
    sha256 = _required_string(values, "sha256")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise MetadataError("Checkpoint metadata field 'sha256' must be lowercase hexadecimal")

    return CheckpointMetadata(
        source_path=_manifest_path(_required_string(values, "source_artifact_path")),
        runtime_path=_manifest_path(_required_string(values, "runtime_artifact_path")),
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _file_identity(path: Path) -> FileIdentity:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_SIZE), b""):
            size_bytes += len(chunk)
            digest.update(chunk)
    return FileIdentity(size_bytes=size_bytes, sha256=digest.hexdigest())


def _require_regular_source(path: Path) -> None:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError as error:
        raise SourceCheckpointError(f"Source checkpoint does not exist: {path}") from error
    except OSError as error:
        raise SourceCheckpointError(f"Could not inspect source checkpoint: {path}") from error
    if not stat.S_ISREG(mode):
        raise SourceCheckpointError(f"Source checkpoint is not a regular file: {path}")


def _verify_identity(
    path: Path,
    expected: CheckpointMetadata,
    *,
    label: str,
) -> FileIdentity:
    try:
        identity = _file_identity(path)
    except OSError as error:
        raise CheckpointVerificationError(f"Could not verify {label} checkpoint: {path}") from error
    if identity.size_bytes != expected.size_bytes:
        raise CheckpointVerificationError(
            f"{label.capitalize()} checkpoint size mismatch: expected "
            f"{expected.size_bytes} bytes, got {identity.size_bytes}"
        )
    if identity.sha256 != expected.sha256:
        raise CheckpointVerificationError(
            f"{label.capitalize()} checkpoint SHA-256 mismatch: expected "
            f"{expected.sha256}, got {identity.sha256}"
        )
    return identity


def _existing_destination_identity(path: Path) -> FileIdentity | None:
    if not os.path.lexists(path):
        return None
    if path.is_symlink():
        raise DestinationConflictError(f"Refusing symbolic-link destination: {path}")
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise DestinationConflictError(f"Could not inspect existing destination: {path}") from error
    if not stat.S_ISREG(mode):
        raise DestinationConflictError(f"Existing destination is not a regular file: {path}")
    try:
        return _file_identity(path)
    except OSError as error:
        raise DestinationConflictError(f"Could not verify existing destination: {path}") from error


def _copy_to_partial(source: Path, destination: Path) -> Path:
    try:
        temporary = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".partial",
            dir=destination.parent,
            delete=False,
        )
    except OSError as error:
        raise InstallationError(
            f"Could not create a partial checkpoint in: {destination.parent}"
        ) from error

    partial_path = Path(temporary.name)
    try:
        with temporary, source.open("rb") as source_handle:
            shutil.copyfileobj(source_handle, temporary, length=COPY_CHUNK_SIZE)
            temporary.flush()
            os.fsync(temporary.fileno())
    except OSError as error:
        try:
            partial_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            raise InstallationError(
                f"Checkpoint copy failed and partial cleanup failed: {partial_path}"
            ) from cleanup_error
        raise InstallationError(
            f"Could not copy checkpoint to partial file: {partial_path}"
        ) from error
    return partial_path


def install_checkpoint(
    *,
    source: str | Path | None = None,
    destination: str | Path | None = None,
    force: bool = False,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> InstallationResult:
    """Verify and atomically install a checkpoint without deserializing it."""

    metadata = load_trusted_metadata(metadata_path)
    source_path = Path(source) if source is not None else metadata.source_path
    destination_path = Path(destination) if destination is not None else metadata.runtime_path

    _require_regular_source(source_path)
    source_identity = _verify_identity(source_path, metadata, label="source")

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise InstallationError(
            f"Could not create destination directory: {destination_path.parent}"
        ) from error
    if not destination_path.parent.is_dir():
        raise InstallationError(
            f"Destination parent is not a directory: {destination_path.parent}"
        )

    existing_identity = _existing_destination_identity(destination_path)
    if existing_identity == source_identity:
        return InstallationResult(
            status="already_installed",
            destination=destination_path,
            identity=existing_identity,
        )
    if existing_identity is not None and not force:
        raise DestinationConflictError(
            f"A different checkpoint already exists at {destination_path}; "
            "use --force to replace it"
        )

    partial_path: Path | None = None
    try:
        partial_path = _copy_to_partial(source_path, destination_path)
        copied_identity = _verify_identity(partial_path, metadata, label="copied")
        try:
            os.replace(partial_path, destination_path)
        except OSError as error:
            raise InstallationError(
                f"Could not atomically install checkpoint at: {destination_path}"
            ) from error
        partial_path = None
    finally:
        if partial_path is not None:
            try:
                partial_path.unlink(missing_ok=True)
            except OSError as error:
                raise InstallationError(
                    f"Could not remove incomplete partial checkpoint: {partial_path}"
                ) from error

    return InstallationResult(
        status="installed",
        destination=destination_path,
        identity=copied_identity,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the local checkpoint installer command-line parser."""

    parser = argparse.ArgumentParser(
        description="Verify and atomically install the local SemiRestore checkpoint."
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="source checkpoint (default: source_artifact_path from checksums.json)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="runtime checkpoint (default: runtime_artifact_path from checksums.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace a different existing regular-file destination",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the installer and translate expected failures into stable exit codes."""

    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        result = install_checkpoint(
            source=args.source,
            destination=args.destination,
            force=args.force,
            metadata_path=metadata_path,
        )
    except InstallerError as error:
        print(f"error: {error}", file=errors)
        return int(error.exit_code)

    action = "already installed" if result.status == "already_installed" else "installed"
    print(
        f"checkpoint {action}: {result.destination} "
        f"({result.identity.size_bytes} bytes, sha256={result.identity.sha256})",
        file=output,
    )
    return int(ExitCode.SUCCESS)


if __name__ == "__main__":
    raise SystemExit(main())
