from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from semirestore import cli
from semirestore.training_checkpoints import TrainingCheckpointError


def _write_pair(root: Path, sample_id: str, split: str) -> dict[str, str]:
    low = root / "data" / "lr" / f"{sample_id}.npy"
    high = root / "data" / "hr" / f"{sample_id}.npy"
    low.parent.mkdir(parents=True, exist_ok=True)
    high.parent.mkdir(parents=True, exist_ok=True)
    np.save(low, np.zeros((6, 6), dtype=np.float32))
    np.save(high, np.zeros((12, 12), dtype=np.float32))
    return {
        "sample_id": sample_id,
        "lr_path": low.relative_to(root / "data").as_posix(),
        "hr_path": high.relative_to(root / "data").as_posix(),
        "split": split,
    }


def _workflow_fixture(tmp_path: Path) -> Path:
    rows = [_write_pair(tmp_path, "train-a", "train"), _write_pair(tmp_path, "val-a", "validation")]
    manifest = tmp_path / "data" / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "lr_path", "hr_path", "split"))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "model": {
            "name": "naf_sr",
            "width": 48,
            "encoder_blocks": [2, 2, 4],
            "middle_blocks": 6,
            "decoder_blocks": [2, 2, 2],
            "dropout": 0.0,
            "statistics_conditioning": True,
            "conditioning_hidden": 64,
        },
        "data": {
            "manifest": "data/manifest.csv",
            "dataset_root": "data",
            "train_split": "train",
            "validation_split": "validation",
        },
        "training": {
            "seed": 2026,
            "device": "cpu",
            "deterministic": True,
            "batch_size": 1,
            "num_workers": 0,
            "max_steps": 2,
            "validation_interval": 1,
            "learning_rate": 0.0002,
            "weight_decay": 0.0,
            "warmup_steps": 1,
            "gradient_clip_norm": 1.0,
            "charbonnier_epsilon": 0.001,
            "d4_augmentation": False,
            "synthetic_probability": 0.0,
            "metric_data_range": 1.0,
            "metric_data_min": 0.0,
            "metric_range_policy": "clip",
            "amp": True,
            "ema_enabled": True,
            "ema_decay": 0.999,
            "best_validation_metric": "psnr_db",
        },
        "output": {"run_dir": "runs/test"},
        "checkpointing": {"keep_last": 2},
    }
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class FakeSummary:
    def as_dict(self) -> dict[str, object]:
        return {"completed_steps": 1, "device": "cpu"}


class FakeTrainer:
    def __init__(self) -> None:
        self.resumed: Path | None = None
        self.model = nn.Conv2d(1, 1, 1)

    def fit(self, train: object, validation: object, *, max_steps: int) -> FakeSummary:
        assert train == "train-loader"
        assert validation == "validation-loader"
        assert max_steps >= 1
        return FakeSummary()

    def resume(self, path: Path) -> None:
        self.resumed = path

    def validate(self, loader: object) -> Any:
        assert loader == "validation-loader"
        return SimpleNamespace(
            as_dict=lambda: {
                "image_count": 1,
                "mean_psnr_db": 20.0,
                "mean_ssim": 0.8,
            }
        )


def test_validate_config_json_is_resolved_and_path_safe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workflow_fixture(tmp_path)

    code = cli.main(["--config", str(config), "--json", "validate-config"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "valid"
    assert payload["resolved_configuration"]["data"]["manifest"] == "data/manifest.csv"
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["checkpoint_identity"]["expected_sha256"].startswith("273abd9d")


def test_dataset_audit_reports_validated_split_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workflow_fixture(tmp_path)

    code = cli.main(["--config", str(config), "--json", "audit-dataset"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["train_samples"] == 1
    assert payload["validation_samples"] == 1
    assert payload["leakage_detected"] is False


def test_dry_run_starts_no_expensive_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workflow_fixture(tmp_path)

    assert cli.main(["--config", str(config), "--json", "dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["expensive_work_started"] is False
    assert payload["seed"] == 2026
    assert payload["planned_max_steps"] == 2


def test_environment_reports_device_seed_and_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workflow_fixture(tmp_path)

    assert cli.main(["--config", str(config), "--json", "environment"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["seed"] == 2026
    assert payload["device_request"] == "cpu"
    assert isinstance(payload["cuda_available"], bool)
    assert payload["checkpoint_identity"]["model_name"] == "naf_sr"


def test_explicit_train_command_uses_bounded_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workflow_fixture(tmp_path)
    fake = FakeTrainer()
    monkeypatch.setattr(
        cli, "_create_training", lambda config: (fake, "train-loader", "validation-loader")
    )

    code = cli.main(["--config", str(config), "--json", "train", "--steps", "1"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["summary"]["completed_steps"] == 1
    assert payload["seed"] == 2026


def test_resume_reports_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workflow_fixture(tmp_path)
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"controlled resume fixture")
    fake = FakeTrainer()
    monkeypatch.setattr(
        cli, "_create_training", lambda config: (fake, "train-loader", "validation-loader")
    )

    code = cli.main(
        [
            "--config",
            str(config),
            "--json",
            "resume",
            "--checkpoint",
            str(checkpoint),
            "--steps",
            "2",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert fake.resumed == checkpoint
    assert payload["resume_checkpoint"] == "last.pt"
    assert len(payload["resume_checkpoint_sha256"]) == 64


def test_reference_evaluation_uses_safe_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workflow_fixture(tmp_path)
    fake = FakeTrainer()
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "checkpoint_role": "best_inference",
            "model_state_dict": fake.model.state_dict(),
        },
        checkpoint,
    )
    monkeypatch.setattr(
        cli, "_create_training", lambda config: (fake, "train-loader", "validation-loader")
    )

    code = cli.main(
        ["--config", str(config), "--json", "evaluate", "--checkpoint", str(checkpoint)]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["reference_metrics"]["mean_psnr_db"] == 20.0
    assert "reference HR" in payload["limitations"][0]


class FakeRestoration:
    png_bytes = b"\x89PNG\r\n\x1a\nfixture"
    media_type = "image/png"
    total_latency_ms = 2.5
    resolved_device = "cpu"
    checkpoint_sha256 = "a" * 64

    def metadata(self) -> dict[str, object]:
        return {"restored_width": 12, "restored_height": 12, "warnings": []}


def test_restore_writes_lossless_png_and_safe_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workflow_fixture(tmp_path)
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"fixture")
    output_path = tmp_path / "restored.png"
    monkeypatch.setattr(
        cli,
        "_restore_once",
        lambda config, args: (
            FakeRestoration(),
            {"checkpoint_path": "semirestore_conditioned.pt", "checkpoint_sha256": "a" * 64},
        ),
    )

    code = cli.main(
        [
            "--config",
            str(config),
            "--json",
            "restore",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output_path.read_bytes() == FakeRestoration.png_bytes
    assert payload["media_type"] == "image/png"
    assert payload["output"] == "restored.png"
    assert str(tmp_path) not in json.dumps(payload)


def test_benchmark_reports_only_measured_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workflow_fixture(tmp_path)
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"fixture")
    report = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        cli,
        "_restore_once",
        lambda config, args: (FakeRestoration(), {"checkpoint_sha256": "a" * 64}),
    )

    code = cli.main(
        [
            "--config",
            str(config),
            "--json",
            "benchmark",
            "--input",
            str(input_path),
            "--iterations",
            "2",
            "--report",
            str(report),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["iterations"] == 2
    assert payload["measured_total_latency_ms"]["mean"] == 2.5
    assert payload["fabricated_values"] is False
    assert json.loads(report.read_text(encoding="utf-8"))["fabricated_values"] is False


def test_human_summary_is_useful_and_compact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workflow_fixture(tmp_path)

    assert cli.main(["--config", str(config), "dry-run"]) == 0
    output = capsys.readouterr().out

    assert "SemiRestore dry-run: ready" in output
    assert "seed: 2026" in output


def test_configuration_error_has_stable_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("model: []\n", encoding="utf-8")

    code = cli.main(["--config", str(config), "validate-config"])

    assert code == 2
    assert "configuration error" in capsys.readouterr().err


def test_checkpoint_and_runtime_failures_have_distinct_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workflow_fixture(tmp_path)

    def checkpoint_failure(config: object, args: object) -> dict[str, object]:
        raise TrainingCheckpointError("safe failure")

    monkeypatch.setitem(cli._COMMANDS, "dry-run", checkpoint_failure)
    assert cli.main(["--config", str(config), "dry-run"]) == 3
    assert "checkpoint error" in capsys.readouterr().err

    def runtime_failure(config: object, args: object) -> dict[str, object]:
        raise RuntimeError("controlled runtime failure")

    monkeypatch.setitem(cli._COMMANDS, "dry-run", runtime_failure)
    assert cli.main(["--config", str(config), "dry-run"]) == 4
    assert "workflow error" in capsys.readouterr().err


def test_cli_module_import_and_help_start_no_training() -> None:
    import_result = subprocess.run(
        [sys.executable, "-c", "import semirestore.cli; print('imported')"],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    help_result = subprocess.run(
        [sys.executable, "scripts/model/workflows.py", "--help"],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert import_result.returncode == 0
    assert import_result.stdout.strip() == "imported"
    assert help_result.returncode == 0
    assert "validate-config" in help_result.stdout
    assert "train" in help_result.stdout


def test_sensitive_mapping_values_are_redacted() -> None:
    payload = cli._redact(
        {"api_token": "do-not-print", "nested": {"password": "also-secret", "seed": 7}}
    )

    rendered = json.dumps(payload)
    assert "do-not-print" not in rendered
    assert "also-secret" not in rendered
    assert payload["nested"]["seed"] == 7
