from __future__ import annotations

import json
from pathlib import Path

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
            "history": [
                {"epoch": 1, "train": {"total": 100.0}, "val": {"total": 80.0}},
                {"epoch": 2, "train": {"total": 30.0}, "val": {"total": 25.0}},
            ],
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

    def fake_export(**kwargs: object) -> Path:
        output_path = Path(str(kwargs["output_path"]))
        output_path.write_text("{}")
        return output_path

    monkeypatch.setattr("tsqbev.overfit.export_nuscenes_predictions", fake_export)
    monkeypatch.setattr(
        "tsqbev.overfit.evaluate_nuscenes_predictions",
        lambda **kwargs: _fake_eval(0.12, 0.02, 0.06),
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
    written = json.loads((artifact_dir / "overfit_gate" / "summary.json").read_text())
    assert written["gate_verdict"]["passed"] is True
