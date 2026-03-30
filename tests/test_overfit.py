from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsqbev.config import ModelConfig
from tsqbev.overfit import run_nuscenes_overfit_gate


def _fake_eval(nds: float, mean_ap: float, car_ap_4m: float) -> dict[str, object]:
    return {
        "mean_ap": mean_ap,
        "nd_score": nds,
        "label_aps": {
            "car": {"0.5": 0.0, "1.0": 0.0, "2.0": 0.0, "4.0": car_ap_4m},
            "truck": {"0.5": 0.0, "1.0": 0.0, "2.0": 0.0, "4.0": 0.0},
        },
    }


def test_run_nuscenes_overfit_gate_writes_summary(monkeypatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    checkpoint = artifact_dir / "overfit_gate" / "checkpoint_last.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint")

    monkeypatch.setattr(
        "tsqbev.overfit.select_nuscenes_subset_tokens",
        lambda *args, **kwargs: [f"tok-{i}" for i in range(32)],
    )
    monkeypatch.setattr(
        "tsqbev.overfit.fit_nuscenes",
        lambda **kwargs: {
            "checkpoint_path": str(checkpoint),
            "selected_checkpoint_path": str(checkpoint),
            "best_checkpoint_path": str(checkpoint),
            "selected_epoch": 2,
            "best_epoch": 2,
            "history": [
                {"epoch": 1, "train": {"total": 100.0}, "val": {"total": 80.0}},
                {"epoch": 2, "train": {"total": 30.0}, "val": {"total": 25.0}},
            ],
            "selected_train": {"total": 30.0},
            "selected_val": {"total": 25.0},
            "best_val": {"total": 25.0},
            "last_train": {"total": 30.0},
            "last_val": {"total": 25.0},
            "epochs": 2,
            "train_steps": 64,
        },
    )
    monkeypatch.setattr(
        "tsqbev.overfit.load_model_from_checkpoint",
        lambda *args, **kwargs: (object(), {}),
    )

    calibration_prediction = (
        artifact_dir / "overfit_gate" / "calibration" / "predictions_s0.05_k32.json"
    )
    monkeypatch.setattr(
        "tsqbev.overfit.export_and_evaluate_nuscenes_grid",
        lambda **kwargs: {
            "selected": {
                "score_threshold": 0.05,
                "top_k": 32,
                "prediction_path": str(calibration_prediction),
                "evaluation": _fake_eval(0.12, 0.02, 0.06),
            },
            "candidates": [],
        },
    )
    monkeypatch.setattr(
        "tsqbev.overfit.prediction_geometry_diagnostics",
        lambda *args, **kwargs: {
            "boxes_per_sample_mean": 12.0,
            "boxes_per_sample_p95": 14.0,
            "boxes_per_sample_max": 16.0,
            "ego_translation_norm_mean": 18.0,
            "ego_translation_norm_p95": 24.0,
            "ego_translation_norm_p99": 30.0,
            "ego_translation_norm_max": 36.0,
        },
    )
    monkeypatch.setattr(
        "tsqbev.overfit.benchmark_forward",
        lambda *args, **kwargs: {"mean_ms": 17.0, "p95_ms": 17.5},
    )

    summary = run_nuscenes_overfit_gate(
        dataroot=tmp_path,
        artifact_dir=artifact_dir,
        config=ModelConfig.rtx5000_nuscenes_query_boost(),
        device="cpu",
    )

    assert summary["status"] == "completed"
    assert summary["gate_verdict"]["passed"] is True
    assert summary["gate_verdict"]["train_total_ratio"] == 0.3
    assert Path(summary["subset_tokens_path"]).exists()
    assert summary["calibration"]["selected"]["top_k"] == 32
    written = json.loads((artifact_dir / "overfit_gate" / "summary.json").read_text())
    assert written["gate_verdict"]["passed"] is True
    assert written["gate_verdict"]["car_ap_4m"] == 0.06


def test_overfit_gate_uses_selected_evaluation_for_car_ap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    checkpoint = artifact_dir / "overfit_gate" / "checkpoint_last.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint")

    monkeypatch.setattr(
        "tsqbev.overfit.select_nuscenes_subset_tokens",
        lambda *args, **kwargs: [f"tok-{i}" for i in range(32)],
    )
    monkeypatch.setattr(
        "tsqbev.overfit.fit_nuscenes",
        lambda **kwargs: {
            "checkpoint_path": str(checkpoint),
            "selected_checkpoint_path": str(checkpoint),
            "best_checkpoint_path": str(checkpoint),
            "selected_epoch": 2,
            "best_epoch": 2,
            "history": [
                {"epoch": 1, "train": {"total": 100.0}, "val": {"total": 80.0}},
                {"epoch": 2, "train": {"total": 30.0}, "val": {"total": 25.0}},
            ],
            "selected_train": {"total": 30.0},
            "selected_val": {"total": 25.0},
            "best_val": {"total": 25.0},
            "last_train": {"total": 30.0},
            "last_val": {"total": 25.0},
            "epochs": 2,
            "train_steps": 64,
        },
    )
    monkeypatch.setattr(
        "tsqbev.overfit.load_model_from_checkpoint",
        lambda *args, **kwargs: (object(), {}),
    )
    monkeypatch.setattr(
        "tsqbev.overfit.export_and_evaluate_nuscenes_grid",
        lambda **kwargs: {
            "selected": {
                "score_threshold": 0.15,
                "top_k": 64,
                "prediction_path": str(
                    artifact_dir / "overfit_gate" / "calibration" / "predictions_s0.15_k64.json"
                ),
                "car_ap_4m": 0.0,
                "evaluation": _fake_eval(0.12, 0.02, 0.06),
            },
            "candidates": [],
        },
    )
    monkeypatch.setattr(
        "tsqbev.overfit.prediction_geometry_diagnostics",
        lambda *args, **kwargs: {
            "boxes_per_sample_mean": 12.0,
            "boxes_per_sample_p95": 14.0,
            "boxes_per_sample_max": 16.0,
            "ego_translation_norm_mean": 18.0,
            "ego_translation_norm_p95": 24.0,
            "ego_translation_norm_p99": 30.0,
            "ego_translation_norm_max": 36.0,
        },
    )
    monkeypatch.setattr(
        "tsqbev.overfit.benchmark_forward",
        lambda *args, **kwargs: {"mean_ms": 17.0, "p95_ms": 17.5},
    )

    summary = run_nuscenes_overfit_gate(
        dataroot=tmp_path,
        artifact_dir=artifact_dir,
        config=ModelConfig.rtx5000_nuscenes_teacher_bootstrap(),
        device="cpu",
    )

    assert summary["gate_verdict"]["car_ap_4m"] == 0.06


def test_run_nuscenes_overfit_gate_finishes_tracker_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    finished: list[str] = []

    class _FakeTracker:
        def log(self, payload: dict[str, object], step: int | None = None) -> None:
            del payload, step

        def summary(self, payload: dict[str, object]) -> None:
            del payload

        def finish(self, *, status: str) -> None:
            finished.append(status)

    monkeypatch.setattr("tsqbev.overfit.start_experiment_tracking", lambda **kwargs: _FakeTracker())
    monkeypatch.setattr(
        "tsqbev.overfit.select_nuscenes_subset_tokens",
        lambda *args, **kwargs: [f"tok-{i}" for i in range(32)],
    )
    monkeypatch.setattr(
        "tsqbev.overfit.fit_nuscenes",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_nuscenes_overfit_gate(
            dataroot=tmp_path,
            artifact_dir=tmp_path / "artifacts",
            config=ModelConfig.small(),
            device="cpu",
        )

    assert finished == ["failed"]
