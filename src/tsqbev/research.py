"""Bounded local research loop for mini nuScenes experiments.

References:
- Karpathy autoresearch workflow template:
  https://github.com/karpathy/autoresearch
- nuScenes official mini split support:
  https://github.com/nutonomy/nuscenes-devkit
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from tsqbev.checkpoints import load_model_from_checkpoint
from tsqbev.config import ModelConfig
from tsqbev.datasets import NuScenesDataset, collate_scene_examples
from tsqbev.eval_nuscenes import evaluate_nuscenes_predictions, export_nuscenes_predictions
from tsqbev.research_guard import ensure_research_loop_enabled
from tsqbev.runtime import benchmark_forward, move_batch, resolve_device
from tsqbev.teacher_backends import TeacherProviderConfig, build_teacher_provider
from tsqbev.teacher_dataset import TeacherAugmentedDataset
from tsqbev.train import fit_nuscenes


@dataclass(slots=True)
class ResearchRecipe:
    """A small bounded experiment configuration."""

    name: str
    note: str
    config: ModelConfig
    batch_size: int
    grad_accum_steps: int
    lr: float = 3e-4
    epochs: int = 6
    num_workers: int = 4
    score_threshold: float = 0.05
    top_k: int = 300


def _mini_recipes() -> list[ResearchRecipe]:
    baseline = ModelConfig.rtx5000_nuscenes_baseline()
    proposal_heavy = baseline.model_copy(
        update={
            "q_lidar": 64,
            "q_2d": 96,
            "q_global": 32,
            "max_object_queries": 96,
            "proposals_per_view": 24,
            "pillar": baseline.pillar.model_copy(update={"q_lidar": 64}),
        }
    )
    efficientnet = proposal_heavy.model_copy(
        update={
            "image_backbone": "efficientnet_b0",
            "pretrained_image_backbone": True,
            "freeze_image_backbone": True,
        }
    )
    unfrozen = proposal_heavy.model_copy(update={"freeze_image_backbone": False})
    return [
        ResearchRecipe(
            name="mini_balanced_mbv3_frozen",
            note="balanced router with frozen MobileNetV3 baseline recipe",
            config=baseline,
            batch_size=2,
            grad_accum_steps=2,
        ),
        ResearchRecipe(
            name="mini_propheavy_mbv3_frozen",
            note=(
                "shift more sparse budget toward camera proposal seeds while keeping "
                "MobileNetV3 frozen"
            ),
            config=proposal_heavy,
            batch_size=2,
            grad_accum_steps=2,
        ),
        ResearchRecipe(
            name="mini_propheavy_effb0_frozen",
            note="proposal-heavy recipe with a stronger frozen EfficientNet-B0 image backbone",
            config=efficientnet,
            batch_size=2,
            grad_accum_steps=2,
            lr=2e-4,
        ),
        ResearchRecipe(
            name="mini_propheavy_mbv3_unfrozen",
            note="proposal-heavy recipe that also finetunes MobileNetV3",
            config=unfrozen,
            batch_size=2,
            grad_accum_steps=2,
            lr=2e-4,
        ),
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True, default=str) for row in rows)
    path.write_text(f"{payload}\n" if payload else "")


@torch.no_grad()
def _measure_source_mix(
    model: torch.nn.Module,
    dataroot: Path,
    *,
    version: str,
    split: str,
    device: str | None,
    teacher_provider_config: TeacherProviderConfig | None = None,
    max_batches: int = 8,
) -> dict[str, float]:
    resolved_device = resolve_device(device)
    dataset: Dataset[Any] = NuScenesDataset(dataroot=dataroot, version=version, split=split)
    if teacher_provider_config is not None:
        dataset = TeacherAugmentedDataset(dataset, build_teacher_provider(teacher_provider_config))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_scene_examples,
        pin_memory=torch.cuda.is_available(),
    )
    model = model.to(resolved_device).eval()
    counts = torch.zeros(3, dtype=torch.float64)
    total_queries = 0.0
    for batch_index, (batch, _metadata) in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_batch(batch, resolved_device)
        outputs = model(batch)
        seed_bank = outputs["seed_bank"]
        source_ids = seed_bank.source_ids.detach().cpu()
        for source_id in range(3):
            counts[source_id] += float((source_ids == source_id).sum())
        total_queries += float(source_ids.numel())
    if total_queries <= 0.0:
        return {"lidar": 0.0, "proposal": 0.0, "global": 0.0}
    return {
        "lidar": float(counts[0] / total_queries),
        "proposal": float(counts[1] / total_queries),
        "global": float(counts[2] / total_queries),
    }


def _select_better_record(
    current_best: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> bool:
    if current_best is None:
        return True
    current_eval = current_best.get("evaluation", {})
    candidate_eval = candidate.get("evaluation", {})
    current_nds = float(current_eval.get("nd_score", float("-inf")))
    candidate_nds = float(candidate_eval.get("nd_score", float("-inf")))
    if candidate_nds != current_nds:
        return candidate_nds > current_nds
    current_map = float(current_eval.get("mean_ap", float("-inf")))
    candidate_map = float(candidate_eval.get("mean_ap", float("-inf")))
    if candidate_map != current_map:
        return candidate_map > current_map
    return float(candidate["val"]["total"]) < float(current_best["val"]["total"])


def run_bounded_research_loop(
    dataroot: str | Path,
    artifact_dir: str | Path,
    *,
    device: str | None = None,
    max_experiments: int = 3,
    teacher_provider_config: TeacherProviderConfig | None = None,
) -> dict[str, Any]:
    """Run a bounded mini-nuScenes experiment sweep and evaluate the best recipe."""

    ensure_research_loop_enabled()
    root = Path(dataroot)
    artifact_root = Path(artifact_dir) / "research_loop"
    artifact_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_recipe: ResearchRecipe | None = None

    for recipe in _mini_recipes()[:max_experiments]:
        run_dir = artifact_root / recipe.name
        record: dict[str, Any] = {
            "recipe": recipe.name,
            "note": recipe.note,
            "status": "started",
            "version": "v1.0-mini",
            "train_split": "mini_train",
            "val_split": "mini_val",
            "batch_size": recipe.batch_size,
            "grad_accum_steps": recipe.grad_accum_steps,
            "lr": recipe.lr,
            "epochs": recipe.epochs,
            "freeze_image_backbone": recipe.config.freeze_image_backbone,
            "image_backbone": recipe.config.image_backbone,
        }
        try:
            train_result = fit_nuscenes(
                dataroot=root,
                artifact_dir=run_dir,
                config=recipe.config,
                version="v1.0-mini",
                train_split="mini_train",
                val_split="mini_val",
                epochs=recipe.epochs,
                lr=recipe.lr,
                grad_accum_steps=recipe.grad_accum_steps,
                batch_size=recipe.batch_size,
                num_workers=recipe.num_workers,
                device=device,
                teacher_provider_config=teacher_provider_config,
                use_amp=False,
                log_every_steps=25,
            )
            bench = benchmark_forward(
                recipe.config,
                steps=10,
                warmup=3,
                batch_size=1,
                device=device,
                image_height=256,
                image_width=704,
            )
            checkpoint_path = Path(str(train_result["checkpoint_path"]))
            model, _ = load_model_from_checkpoint(checkpoint_path)
            prediction_path = run_dir / "mini_predictions.json"
            export_nuscenes_predictions(
                model=model,
                dataroot=root,
                version="v1.0-mini",
                split="mini_val",
                output_path=prediction_path,
                score_threshold=recipe.score_threshold,
                top_k=recipe.top_k,
                device=device,
                teacher_provider_config=teacher_provider_config,
            )
            evaluation = evaluate_nuscenes_predictions(
                dataroot=root,
                version="v1.0-mini",
                split="mini_val",
                result_path=prediction_path,
                output_dir=run_dir / "mini_eval",
            )
            source_mix = _measure_source_mix(
                model,
                root,
                version="v1.0-mini",
                split="mini_val",
                device=device,
                teacher_provider_config=teacher_provider_config,
            )
            record.update(
                {
                    "status": "completed",
                    "train": train_result["last_train"],
                    "val": train_result["last_val"],
                    "benchmark": bench,
                    "checkpoint_path": str(checkpoint_path),
                    "prediction_path": str(prediction_path),
                    "evaluation": evaluation,
                    "source_mix": source_mix,
                    "score_threshold": recipe.score_threshold,
                    "top_k": recipe.top_k,
                    "teacher_provider": (
                        teacher_provider_config.kind
                        if teacher_provider_config is not None
                        else None
                    ),
                }
            )
            if _select_better_record(best_record, record):
                record["decision"] = "keep"
                best_record = record
                best_recipe = recipe
            else:
                record["decision"] = "discard"
        except Exception as exc:
            record.update({"status": "error", "error": repr(exc), "decision": "discard"})
        records.append(record)
        _write_jsonl(artifact_root / "results.jsonl", records)

    if best_record is None or best_recipe is None:
        failed_summary: dict[str, Any] = {
            "status": "failed",
            "records_path": str(artifact_root / "results.jsonl"),
            "records": records,
        }
        (artifact_root / "summary.json").write_text(
            json.dumps(failed_summary, indent=2, default=str)
        )
        return failed_summary

    summary: dict[str, Any] = {
        "status": "completed",
        "selected_recipe": best_recipe.name,
        "records_path": str(artifact_root / "results.jsonl"),
        "selected_checkpoint": best_record["checkpoint_path"],
        "selected_record": best_record,
        "evaluation": best_record["evaluation"],
        "recipes": [
            {
                "name": recipe.name,
                "note": recipe.note,
                "config": recipe.config.model_dump(),
                "batch_size": recipe.batch_size,
                "grad_accum_steps": recipe.grad_accum_steps,
                "lr": recipe.lr,
                "epochs": recipe.epochs,
                "num_workers": recipe.num_workers,
                "score_threshold": recipe.score_threshold,
                "top_k": recipe.top_k,
            }
            for recipe in _mini_recipes()[:max_experiments]
        ],
    }
    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
