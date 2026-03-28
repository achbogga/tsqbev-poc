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

from tsqbev.checkpoints import load_model_from_checkpoint
from tsqbev.config import ModelConfig
from tsqbev.eval_nuscenes import evaluate_nuscenes_predictions, export_nuscenes_predictions
from tsqbev.research_guard import ensure_research_loop_enabled
from tsqbev.runtime import benchmark_forward
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
    epochs: int = 1
    num_workers: int = 4


def _mini_recipes() -> list[ResearchRecipe]:
    baseline = ModelConfig.rtx5000_nuscenes_baseline()
    batch4 = baseline.model_copy()
    unfrozen = baseline.model_copy(update={"freeze_image_backbone": False})
    return [
        ResearchRecipe(
            name="mini_mbv3_frozen_bs2",
            note="baseline frozen MobileNetV3 with conservative batching",
            config=baseline,
            batch_size=2,
            grad_accum_steps=2,
        ),
        ResearchRecipe(
            name="mini_mbv3_frozen_bs4",
            note="use more VRAM by increasing batch size while keeping the backbone frozen",
            config=batch4,
            batch_size=4,
            grad_accum_steps=1,
        ),
        ResearchRecipe(
            name="mini_mbv3_unfrozen_bs2",
            note="allow backbone finetuning at the original conservative batch size",
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


def run_bounded_research_loop(
    dataroot: str | Path,
    artifact_dir: str | Path,
    *,
    device: str | None = None,
    max_experiments: int = 3,
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
            record.update(
                {
                    "status": "completed",
                    "train": train_result["last_train"],
                    "val": train_result["last_val"],
                    "benchmark": bench,
                    "checkpoint_path": train_result["checkpoint_path"],
                }
            )
            val_total = float(record["val"]["total"])
            if best_record is None or val_total < float(best_record["val"]["total"]):
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

    model, _ = load_model_from_checkpoint(best_record["checkpoint_path"])
    prediction_path = artifact_root / "best_mini_predictions.json"
    export_nuscenes_predictions(
        model=model,
        dataroot=root,
        version="v1.0-mini",
        split="mini_val",
        output_path=prediction_path,
        device=device,
    )
    evaluation = evaluate_nuscenes_predictions(
        dataroot=root,
        version="v1.0-mini",
        split="mini_val",
        result_path=prediction_path,
        output_dir=artifact_root / "mini_eval",
    )

    summary: dict[str, Any] = {
        "status": "completed",
        "selected_recipe": best_recipe.name,
        "records_path": str(artifact_root / "results.jsonl"),
        "selected_checkpoint": best_record["checkpoint_path"],
        "selected_record": best_record,
        "evaluation": evaluation,
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
            }
            for recipe in _mini_recipes()
        ],
    }
    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
