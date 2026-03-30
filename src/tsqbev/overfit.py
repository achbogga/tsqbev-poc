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
from typing import Any, Literal, cast

from tsqbev.checkpoints import load_model_from_checkpoint
from tsqbev.config import ModelConfig
from tsqbev.datasets import NuScenesDataset
from tsqbev.eval_nuscenes import (
    export_and_evaluate_nuscenes_grid,
    prediction_geometry_diagnostics,
)
from tsqbev.runtime import benchmark_forward
from tsqbev.teacher_backends import TeacherProviderConfig
from tsqbev.tracking import TrackingMetadata, start_experiment_tracking
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


def _metric_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _root_cause_verdict(
    *,
    train_ratio: float,
    nds: float,
    mean_ap: float,
    car_ap_4m: float,
    nonzero_classes: int,
    boxes_per_sample_mean: float,
    boxes_per_sample_p95: float,
) -> str:
    if train_ratio > 0.4:
        return (
            "optimization bottleneck: the subset still does not overfit enough; use overfit-mode "
            "schedule/regularization and checkpoint selection before scaling search"
        )
    if boxes_per_sample_mean > 40.0 or boxes_per_sample_p95 > 60.0:
        return (
            "ranking bottleneck: query export is still overproducing boxes; tighten "
            "objectness/class "
            "calibration and unmatched negative handling"
        )
    if car_ap_4m <= 0.0 and nonzero_classes <= 2:
        return (
            "vehicle-emergence bottleneck: the model is learning easy classes before cars; "
            "bias the "
            "next loop toward teacher-seeded vehicle grounding and ranking calibration"
        )
    if nds < 0.05 or mean_ap < 0.01:
        return (
            "generalization bottleneck: detection is improving but not enough; keep teacher-seeded "
            "overfit recovery as the incumbent before broader mini sweeps"
        )
    return (
        "near-pass: continue with paired mini-val exploitation around the repaired "
        "overfit incumbent"
    )


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
    optimizer_schedule: Literal["cosine", "constant"] = "constant",
    grad_clip_norm: float | None = 5.0,
    keep_best_checkpoint: bool = True,
    loss_mode: Literal["baseline", "focal_hardneg"] = "baseline",
    hard_negative_ratio: int = 3,
    hard_negative_cap: int = 96,
    score_threshold_candidates: tuple[float, ...] = (0.05, 0.15, 0.25),
    top_k_candidates: tuple[int, ...] = (32, 64, 112),
) -> dict[str, Any]:
    """Run the 32-sample overfit gate and write a machine-readable verdict."""

    root = Path(dataroot)
    artifact_root = Path(artifact_dir) / "overfit_gate"
    artifact_root.mkdir(parents=True, exist_ok=True)
    tracker = start_experiment_tracking(
        artifact_dir=artifact_root,
        config=config,
        metadata=TrackingMetadata(
            suite="gate",
            dataset="nuscenes",
            job_type="overfit-nuscenes",
            run_name=f"overfit-{config.image_backbone}",
            group=f"overfit-{version}",
            tags=(
                "overfit-gate",
                version,
                split,
                config.image_backbone,
                config.teacher_seed_mode,
            ),
            extra_config={
                "split": split,
                "subset_size": subset_size,
                "scene_name": scene_name,
                "teacher_provider": (
                    teacher_provider_config.kind
                    if teacher_provider_config is not None
                    else None
                ),
            },
        ),
        config_payload={
            "model": config.model_dump(),
            "train": {
                "epochs": epochs,
                "max_train_steps": max_train_steps,
                "lr": lr,
                "weight_decay": weight_decay,
                "grad_accum_steps": grad_accum_steps,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "optimizer_schedule": optimizer_schedule,
                "grad_clip_norm": grad_clip_norm,
                "keep_best_checkpoint": keep_best_checkpoint,
                "loss_mode": loss_mode,
                "hard_negative_ratio": hard_negative_ratio,
                "hard_negative_cap": hard_negative_cap,
                "score_threshold_candidates": list(score_threshold_candidates),
                "top_k_candidates": list(top_k_candidates),
            },
        },
    )
    status = "failed"
    try:
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
            optimizer_schedule=optimizer_schedule,
            grad_clip_norm=grad_clip_norm,
            keep_best_checkpoint=keep_best_checkpoint,
            loss_mode=loss_mode,
            hard_negative_ratio=hard_negative_ratio,
            hard_negative_cap=hard_negative_cap,
            tracker=tracker,
        )
        checkpoint_path = Path(str(train_result["checkpoint_path"]))
        model, _ = load_model_from_checkpoint(checkpoint_path)
        calibration = export_and_evaluate_nuscenes_grid(
            model=model,
            dataroot=root,
            version=version,
            split=split,
            output_dir=artifact_root / "calibration",
            score_threshold_candidates=score_threshold_candidates,
            top_k_candidates=top_k_candidates,
            device=device,
            sample_tokens=subset_tokens,
            teacher_provider_config=teacher_provider_config,
        )
        selected_calibration = calibration["selected"]
        assert isinstance(selected_calibration, dict)
        evaluation = selected_calibration["evaluation"]
        assert isinstance(evaluation, dict)
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
        prediction_path = Path(str(selected_calibration["prediction_path"]))
        prediction_geometry = prediction_geometry_diagnostics(
            prediction_path,
            dataroot=root,
            version=version,
        )
        boxes_per_sample_mean = float(prediction_geometry["boxes_per_sample_mean"])
        boxes_per_sample_p95 = float(prediction_geometry["boxes_per_sample_p95"])
        nonzero_classes = _count_nonzero_classes(label_aps)

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
            "nonzero_classes": nonzero_classes,
            "car_ap_4m": car_ap_4m,
        }
        summary = {
            "status": "completed",
            "subset_size": subset_size,
            "scene_name": scene_name,
            "subset_tokens_path": str(artifact_root / "subset_tokens.json"),
            "checkpoint_path": str(checkpoint_path),
            "prediction_path": str(prediction_path),
            "selected_checkpoint_path": str(train_result["selected_checkpoint_path"]),
            "best_checkpoint_path": str(train_result["best_checkpoint_path"]),
            "best_epoch": _metric_int(train_result.get("best_epoch")),
            "selected_epoch": _metric_int(train_result.get("selected_epoch")),
            "train": {
                "epochs": train_result["epochs"],
                "train_steps": train_result["train_steps"],
                "initial_train_total": initial_train_total,
                "final_train_total": final_train_total,
                "final_val_total": float(last_val["total"]),
                "selected_train_total": float(
                    cast(dict[str, Any], train_result["selected_train"])["total"]
                ),
                "selected_val_total": float(
                    cast(dict[str, Any], train_result["selected_val"])["total"]
                ),
                "best_val_total": float(cast(dict[str, Any], train_result["best_val"])["total"])
                if train_result.get("best_val") is not None
                else float(last_val["total"]),
            },
            "evaluation": evaluation,
            "calibration": calibration,
            "prediction_geometry": prediction_geometry,
            "benchmark": benchmark,
            "gate_verdict": verdict,
            "root_cause_verdict": _root_cause_verdict(
                train_ratio=train_ratio,
                nds=nds,
                mean_ap=mean_ap,
                car_ap_4m=car_ap_4m,
                nonzero_classes=nonzero_classes,
                boxes_per_sample_mean=boxes_per_sample_mean,
                boxes_per_sample_p95=boxes_per_sample_p95,
            ),
        }
        (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        epochs_run = _metric_int(train_result.get("epochs", 0))
        tracker.log(
            {
                "epoch": epochs_run,
                "gate_train_total_ratio": train_ratio,
                "gate_official_nds": nds,
                "gate_official_map": mean_ap,
                "gate_car_ap_4m": car_ap_4m,
                "gate_nonzero_classes": verdict["nonzero_classes"],
                "benchmark_mean_ms": float(benchmark["mean_ms"]),
                "benchmark_p95_ms": float(benchmark["p95_ms"]),
            },
            step=epochs_run,
        )
        tracker.summary(
            {
                "subset_size": subset_size,
                "scene_name": scene_name,
                "subset_tokens_path": str(artifact_root / "subset_tokens.json"),
                "checkpoint_path": str(checkpoint_path),
                "prediction_path": str(prediction_path),
                "gate_passed": verdict["passed"],
                "gate_train_total_ratio": train_ratio,
                "gate_official_nds": nds,
                "gate_official_map": mean_ap,
                "gate_car_ap_4m": car_ap_4m,
                "benchmark_mean_ms": float(benchmark["mean_ms"]),
                "benchmark_p95_ms": float(benchmark["p95_ms"]),
            }
        )
        status = "completed"
        return summary
    finally:
        tracker.finish(status=status)
