"""Tiny-subset overfit gate for nuScenes mini.

References:
- nuScenes detection evaluation:
  https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py
- Scale gate contract:
  ../specs/005-scale-gate-contract.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from tsqbev.checkpoints import load_model_from_checkpoint
from tsqbev.config import ModelConfig
from tsqbev.datasets import NuScenesDataset
from tsqbev.eval_nuscenes import evaluate_nuscenes_predictions, export_nuscenes_predictions
from tsqbev.runtime import benchmark_forward
from tsqbev.teacher_backends import TeacherProviderConfig
from tsqbev.train import fit_nuscenes


def select_nuscenes_subset_tokens(
    dataroot: str | Path,
    *,
    version: str,
    split: str,
    subset_size: int,
    scene_name: str | None = None,
) -> list[str]:
    """Select a deterministic fixed subset of nuScenes sample tokens."""

    dataset = NuScenesDataset(dataroot=dataroot, version=version, split=split)
    if scene_name is None:
        tokens = dataset.sample_tokens
    else:
        tokens = [
            token
            for token in dataset.sample_tokens
            if dataset.nusc.get(
                "scene", dataset.nusc.get("sample", token)["scene_token"]
            )["name"]
            == scene_name
        ]
    if len(tokens) < subset_size:
        raise ValueError(
            f"requested {subset_size} tokens from split={split} scene={scene_name}, "
            f"only {len(tokens)} available"
        )
    return tokens[:subset_size]


def _count_nonzero_classes(label_aps: dict[str, Any]) -> int:
    count = 0
    for distance_map in label_aps.values():
        if isinstance(distance_map, dict) and any(
            float(value) > 0.0 for value in distance_map.values()
        ):
            count += 1
    return count


def run_nuscenes_overfit_gate(
    dataroot: str | Path,
    artifact_dir: str | Path,
    *,
    config: ModelConfig,
    version: str = "v1.0-mini",
    split: str = "mini_train",
    subset_size: int = 32,
    scene_name: str | None = None,
    epochs: int = 128,
    max_train_steps: int = 1024,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_accum_steps: int = 1,
    batch_size: int = 4,
    num_workers: int = 4,
    device: str | None = None,
    teacher_provider_config: TeacherProviderConfig | None = None,
) -> dict[str, Any]:
    """Run the 32-sample overfit gate and write a machine-readable verdict."""

    root = Path(dataroot)
    artifact_root = Path(artifact_dir) / "overfit_gate"
    artifact_root.mkdir(parents=True, exist_ok=True)
    subset_tokens = select_nuscenes_subset_tokens(
        root,
        version=version,
        split=split,
        subset_size=subset_size,
        scene_name=scene_name,
    )
    (artifact_root / "subset_tokens.json").write_text(json.dumps(subset_tokens, indent=2))

    train_result = fit_nuscenes(
        dataroot=root,
        artifact_dir=artifact_root,
        config=config,
        version=version,
        train_split=split,
        val_split=split,
        train_sample_tokens=subset_tokens,
        val_sample_tokens=subset_tokens,
        epochs=epochs,
        max_train_steps=max_train_steps,
        lr=lr,
        weight_decay=weight_decay,
        grad_accum_steps=grad_accum_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        teacher_provider_config=teacher_provider_config,
        use_amp=False,
        log_every_steps=25,
    )
    checkpoint_path = Path(str(train_result["checkpoint_path"]))
    model, _ = load_model_from_checkpoint(checkpoint_path)
    prediction_path = artifact_root / "predictions_subset.json"
    export_nuscenes_predictions(
        model=model,
        dataroot=root,
        version=version,
        split=split,
        output_path=prediction_path,
        score_threshold=0.05,
        top_k=300,
        device=device,
        sample_tokens=subset_tokens,
        teacher_provider_config=teacher_provider_config,
    )
    evaluation = evaluate_nuscenes_predictions(
        dataroot=root,
        version=version,
        split=split,
        result_path=prediction_path,
        output_dir=artifact_root / "eval",
        sample_tokens=subset_tokens,
    )
    benchmark = benchmark_forward(
        config,
        steps=10,
        warmup=3,
        batch_size=1,
        device=device,
        image_height=256,
        image_width=704,
    )

    history = train_result["history"]
    assert isinstance(history, list) and history
    first_history = cast(dict[str, Any], history[0])
    first_train = cast(dict[str, Any], first_history["train"])
    last_train = cast(dict[str, Any], train_result["last_train"])
    last_val = cast(dict[str, Any], train_result["last_val"])
    initial_train_total = float(first_train["total"])
    final_train_total = float(last_train["total"])
    train_ratio = (
        final_train_total / initial_train_total
        if initial_train_total > 0.0
        else float("inf")
    )

    label_aps = evaluation.get("label_aps", {})
    assert isinstance(label_aps, dict)
    car_distance_aps = label_aps.get("car", {})
    if not isinstance(car_distance_aps, dict):
        car_distance_aps = {}
    car_ap_4m = float(cast(float, car_distance_aps.get("4.0", 0.0)))
    nds = float(cast(float, evaluation.get("nd_score", 0.0)))
    mean_ap = float(cast(float, evaluation.get("mean_ap", 0.0)))

    verdict = {
        "passed": (
            train_ratio <= 0.4
            and nds >= 0.10
            and mean_ap > 0.0
            and car_ap_4m > 0.0
        ),
        "train_total_ratio": train_ratio,
        "official_nds": nds,
        "official_map": mean_ap,
        "nonzero_classes": _count_nonzero_classes(label_aps),
        "car_ap_4m": car_ap_4m,
    }
    summary = {
        "status": "completed",
        "subset_size": subset_size,
        "scene_name": scene_name,
        "subset_tokens_path": str(artifact_root / "subset_tokens.json"),
        "checkpoint_path": str(checkpoint_path),
        "prediction_path": str(prediction_path),
        "train": {
            "epochs": train_result["epochs"],
            "train_steps": train_result["train_steps"],
            "initial_train_total": initial_train_total,
            "final_train_total": final_train_total,
            "final_val_total": float(last_val["total"]),
        },
        "evaluation": evaluation,
        "benchmark": benchmark,
        "gate_verdict": verdict,
    }
    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
