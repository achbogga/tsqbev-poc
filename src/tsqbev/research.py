"""Bounded local research loop for mini nuScenes experiments.

References:
- Karpathy autoresearch workflow template:
  https://github.com/karpathy/autoresearch
- Karpathy autoresearch baseline program:
  https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md
- nuScenes official mini split support:
  https://github.com/nutonomy/nuscenes-devkit
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

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
from tsqbev.tracking import TrackingMetadata, start_experiment_tracking
from tsqbev.train import fit_nuscenes

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TSV_HEADER = [
    "run_id",
    "recipe",
    "stage",
    "parent_recipe",
    "teacher_seed_mode",
    "teacher_provider",
    "status",
    "interim_decision",
    "final_decision",
    "best_so_far",
    "nds",
    "map",
    "val_total",
    "latency_ms",
    "lidar_share",
    "proposal_share",
    "global_share",
    "hypothesis",
    "mutation_reason",
    "decision_reason",
    "checkpoint_path",
]


@dataclass(slots=True)
class ResearchRecipe:
    """A bounded experiment configuration plus its research rationale."""

    name: str
    note: str
    hypothesis: str
    mutation_reason: str
    config: ModelConfig
    stage: Literal["baseline", "explore", "exploit"] = "explore"
    parent_recipe: str | None = None
    use_teacher_provider: bool = False
    batch_size: int = 2
    grad_accum_steps: int = 2
    lr: float = 3e-4
    epochs: int = 6
    max_train_steps: int | None = 960
    num_workers: int = 4
    score_threshold: float = 0.05
    top_k: int = 300


def _baseline_recipe() -> ResearchRecipe:
    baseline = ModelConfig.rtx5000_nuscenes_baseline()
    return ResearchRecipe(
        name="mini_balanced_mbv3_frozen",
        note="balanced router with frozen MobileNetV3 baseline recipe",
        hypothesis=(
            "a small pretrained image encoder plus balanced tri-source routing should "
            "establish the non-collapsed mini baseline"
        ),
        mutation_reason="baseline reference recipe",
        config=baseline,
        stage="baseline",
    )


def _proposal_heavy_recipe() -> ResearchRecipe:
    baseline = ModelConfig.rtx5000_nuscenes_baseline()
    config = _updated_config(
        baseline,
        q_lidar=64,
        q_2d=96,
        q_global=32,
        max_object_queries=96,
        proposals_per_view=24,
    )
    return ResearchRecipe(
        name="mini_propheavy_mbv3_frozen",
        note="shift more sparse budget toward camera proposal seeds",
        hypothesis=(
            "more camera proposal capacity should improve recall once the router stops "
            "collapsing to near-ego LiDAR seeds"
        ),
        mutation_reason="increase proposal-seed share while keeping the pretrained backbone fixed",
        config=config,
        stage="explore",
        parent_recipe="mini_balanced_mbv3_frozen",
    )


def _efficientnet_recipe(parent: ResearchRecipe) -> ResearchRecipe:
    config = parent.config.model_copy(
        update={
            "image_backbone": "efficientnet_b0",
            "pretrained_image_backbone": True,
            "freeze_image_backbone": True,
        }
    )
    return ResearchRecipe(
        name="mini_propheavy_effb0_frozen",
        note="proposal-heavy recipe with a stronger frozen EfficientNet-B0 image backbone",
        hypothesis=(
            "a slightly stronger pretrained image encoder may raise proposal quality "
            "without breaking the deployment budget"
        ),
        mutation_reason="swap the image backbone while holding the sparse-query budget fixed",
        config=config,
        stage="explore",
        parent_recipe=parent.name,
        batch_size=parent.batch_size,
        grad_accum_steps=parent.grad_accum_steps,
        lr=2e-4,
        epochs=parent.epochs,
        num_workers=parent.num_workers,
        score_threshold=parent.score_threshold,
        top_k=parent.top_k,
    )


def _updated_config(
    config: ModelConfig,
    *,
    image_backbone: str | None = None,
    pretrained_image_backbone: bool | None = None,
    freeze_image_backbone: bool | None = None,
    q_lidar: int | None = None,
    q_2d: int | None = None,
    q_global: int | None = None,
    max_object_queries: int | None = None,
    proposals_per_view: int | None = None,
    teacher_seed_mode: str | None = None,
) -> ModelConfig:
    updates: dict[str, object] = {}
    pillar_updates: dict[str, object] = {}
    if image_backbone is not None:
        updates["image_backbone"] = image_backbone
    if pretrained_image_backbone is not None:
        updates["pretrained_image_backbone"] = pretrained_image_backbone
    if freeze_image_backbone is not None:
        updates["freeze_image_backbone"] = freeze_image_backbone
    if q_lidar is not None:
        updates["q_lidar"] = q_lidar
        pillar_updates["q_lidar"] = q_lidar
    if q_2d is not None:
        updates["q_2d"] = q_2d
    if q_global is not None:
        updates["q_global"] = q_global
    if max_object_queries is not None:
        updates["max_object_queries"] = max_object_queries
    if proposals_per_view is not None:
        updates["proposals_per_view"] = proposals_per_view
    if teacher_seed_mode is not None:
        updates["teacher_seed_mode"] = teacher_seed_mode
    if pillar_updates:
        updates["pillar"] = config.pillar.model_copy(update=pillar_updates)
    return config.model_copy(update=updates)


def _clone_recipe(
    recipe: ResearchRecipe,
    *,
    name: str,
    note: str,
    hypothesis: str,
    mutation_reason: str,
    config: ModelConfig | None = None,
    stage: Literal["baseline", "explore", "exploit"] = "exploit",
    parent_recipe: str | None = None,
    use_teacher_provider: bool | None = None,
    batch_size: int | None = None,
    grad_accum_steps: int | None = None,
    lr: float | None = None,
    epochs: int | None = None,
    max_train_steps: int | None = None,
    num_workers: int | None = None,
    score_threshold: float | None = None,
    top_k: int | None = None,
) -> ResearchRecipe:
    return ResearchRecipe(
        name=name,
        note=note,
        hypothesis=hypothesis,
        mutation_reason=mutation_reason,
        config=config if config is not None else recipe.config,
        stage=stage,
        parent_recipe=recipe.name if parent_recipe is None else parent_recipe,
        use_teacher_provider=(
            recipe.use_teacher_provider if use_teacher_provider is None else use_teacher_provider
        ),
        batch_size=recipe.batch_size if batch_size is None else batch_size,
        grad_accum_steps=recipe.grad_accum_steps if grad_accum_steps is None else grad_accum_steps,
        lr=recipe.lr if lr is None else lr,
        epochs=recipe.epochs if epochs is None else epochs,
        max_train_steps=recipe.max_train_steps if max_train_steps is None else max_train_steps,
        num_workers=recipe.num_workers if num_workers is None else num_workers,
        score_threshold=recipe.score_threshold if score_threshold is None else score_threshold,
        top_k=recipe.top_k if top_k is None else top_k,
    )


def _load_previous_incumbent(artifact_root: Path) -> ResearchRecipe | None:
    summary_path = artifact_root / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None
    selected = summary.get("selected_record")
    if not isinstance(selected, dict):
        return None
    config_payload = selected.get("config")
    if not isinstance(config_payload, dict):
        return None
    try:
        config = ModelConfig.model_validate(config_payload)
    except Exception:
        return None
    previous_name = str(selected.get("recipe", "prior_incumbent"))
    return ResearchRecipe(
        name=f"carryover_{previous_name}",
        note="recheck the last promoted local incumbent before new mutations",
        hypothesis=(
            "the previously promoted mini recipe should reproduce before spending more "
            "budget on follow-up mutations"
        ),
        mutation_reason="carry forward the previous summary winner as the new invocation baseline",
        config=config,
        stage="baseline",
        parent_recipe=previous_name,
        use_teacher_provider=bool(selected.get("use_teacher_provider", False)),
        batch_size=int(selected.get("batch_size", 2)),
        grad_accum_steps=int(selected.get("grad_accum_steps", 2)),
        lr=float(selected.get("lr", 3e-4)),
        epochs=int(selected.get("epochs", 6)),
        max_train_steps=(
            int(selected["max_train_steps"])
            if selected.get("max_train_steps") is not None
            else 960
        ),
        num_workers=int(selected.get("num_workers", 4)),
        score_threshold=float(selected.get("score_threshold", 0.05)),
        top_k=int(selected.get("top_k", 300)),
    )


def _make_teacher_kd_recipe(
    recipe: ResearchRecipe,
    *,
    stage: Literal["baseline", "explore", "exploit"] = "explore",
) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_teacher_kd",
        note="turn on cached teacher-guided supervision without changing seed geometry",
        hypothesis=(
            "score-weighted teacher box and class supervision should improve ranking and "
            "localization before more invasive seed mutations"
        ),
        mutation_reason="enable teacher cache supervision as a paired ablation",
        stage=stage,
        use_teacher_provider=True,
    )


def _initial_recipes(
    artifact_root: Path,
    *,
    teacher_provider_available: bool = False,
) -> list[ResearchRecipe]:
    carryover = _load_previous_incumbent(artifact_root)
    if carryover is not None:
        query_boost = _make_query_boost_recipe(
            carryover,
            source_mix={"lidar": 0.33, "proposal": 0.50, "global": 0.17},
            stage="explore",
        )
        lr_down = _make_lr_down_recipe(carryover, stage="explore")
        recipes = [carryover]
        if teacher_provider_available:
            recipes.append(_make_teacher_kd_recipe(carryover, stage="explore"))
        recipes.extend([query_boost, lr_down])
        return recipes
    baseline = _baseline_recipe()
    proposal = _proposal_heavy_recipe()
    efficientnet = _efficientnet_recipe(proposal)
    recipes = [baseline, proposal, efficientnet]
    if teacher_provider_available:
        recipes.insert(1, _make_teacher_kd_recipe(baseline, stage="explore"))
    return recipes


def _make_lr_down_recipe(
    recipe: ResearchRecipe,
    *,
    stage: Literal["baseline", "explore", "exploit"] = "exploit",
) -> ResearchRecipe:
    lr = max(recipe.lr * 0.67, 1e-4)
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_lr_down",
        note="reduce the learning rate for a more stable late-stage fit",
        hypothesis=(
            "the current incumbent may be overshooting boxes and class scores; a lower "
            "learning rate may improve ranking and localization"
        ),
        mutation_reason="decrease the learning rate by one conservative step",
        stage=stage,
        lr=lr,
    )


def _make_query_boost_recipe(
    recipe: ResearchRecipe,
    *,
    source_mix: dict[str, float],
    stage: Literal["baseline", "explore", "exploit"] = "exploit",
) -> ResearchRecipe:
    proposal_share = float(source_mix.get("proposal", 0.0))
    lidar_share = float(source_mix.get("lidar", 0.0))
    q_lidar = recipe.config.q_lidar
    q_2d = recipe.config.q_2d
    if proposal_share >= lidar_share:
        q_2d = min(q_2d + 16, 128)
    else:
        q_lidar = min(q_lidar + 16, 128)
    max_object_queries = min(
        recipe.config.max_object_queries + 16,
        q_lidar + q_2d + recipe.config.q_global,
    )
    config = _updated_config(
        recipe.config,
        q_lidar=q_lidar,
        q_2d=q_2d,
        max_object_queries=max_object_queries,
        proposals_per_view=min(recipe.config.proposals_per_view + 8, 32),
    )
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_query_boost",
        note="increase retained sparse object capacity around the incumbent source mix",
        hypothesis=(
            "a small query-budget increase around the current dominant source should "
            "improve recall without collapsing the routed bank"
        ),
        mutation_reason="add one bounded sparse-budget increment around the incumbent source mix",
        config=config,
        stage=stage,
    )


def _make_teacher_seed_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    config = _updated_config(recipe.config, teacher_seed_mode="replace_lidar_refs")
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_teacher_ref_seed",
        note="replace LiDAR reference centers with cached external teacher centers",
        hypothesis=(
            "a strong external LiDAR teacher should improve geometric grounding without "
            "discarding the student's learned LiDAR query embeddings"
        ),
        mutation_reason=(
            "inject teacher geometry into the LiDAR reference path as a paired ablation"
        ),
        config=config,
        stage="exploit",
        use_teacher_provider=True,
    )


def _make_unfreeze_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    config = _updated_config(recipe.config, freeze_image_backbone=False)
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_unfreeze",
        note="unfreeze the pretrained image backbone",
        hypothesis=(
            "the current frozen backbone may be bottlenecking proposal quality on the "
            "mini split; careful finetuning may help"
        ),
        mutation_reason="unfreeze the pretrained image backbone as a higher-variance follow-up",
        config=config,
        stage="exploit",
        lr=min(recipe.lr, 2e-4),
    )


def _build_exploitation_recipes(
    incumbent_recipe: ResearchRecipe,
    incumbent_record: dict[str, Any],
    teacher_provider_config: TeacherProviderConfig | None,
    remaining_budget: int,
) -> list[ResearchRecipe]:
    if remaining_budget <= 0:
        return []
    source_mix = incumbent_record.get("source_mix", {})
    assert isinstance(source_mix, dict)
    candidates = [
        _make_query_boost_recipe(incumbent_recipe, source_mix=source_mix),
        _make_lr_down_recipe(incumbent_recipe),
    ]
    if teacher_provider_config is not None:
        if not incumbent_recipe.use_teacher_provider:
            candidates.insert(0, _make_teacher_kd_recipe(incumbent_recipe, stage="exploit"))
        elif incumbent_recipe.config.teacher_seed_mode == "off":
            candidates.insert(0, _make_teacher_seed_recipe(incumbent_recipe))
    if incumbent_recipe.config.freeze_image_backbone:
        candidates.append(_make_unfreeze_recipe(incumbent_recipe))
    deduped: list[ResearchRecipe] = []
    seen_names: set[str] = set()
    for candidate in candidates:
        if candidate.name in seen_names:
            continue
        seen_names.add(candidate.name)
        deduped.append(candidate)
    return deduped[:remaining_budget]


def _serialize_recipe(recipe: ResearchRecipe) -> dict[str, Any]:
    return {
        "recipe": recipe.name,
        "stage": recipe.stage,
        "parent_recipe": recipe.parent_recipe,
        "use_teacher_provider": recipe.use_teacher_provider,
        "note": recipe.note,
        "hypothesis": recipe.hypothesis,
        "mutation_reason": recipe.mutation_reason,
        "config": recipe.config.model_dump(),
        "batch_size": recipe.batch_size,
        "grad_accum_steps": recipe.grad_accum_steps,
        "lr": recipe.lr,
        "epochs": recipe.epochs,
        "max_train_steps": recipe.max_train_steps,
        "num_workers": recipe.num_workers,
        "score_threshold": recipe.score_threshold,
        "top_k": recipe.top_k,
    }


def _warm_start_checkpoint_for_recipe(
    recipe: ResearchRecipe,
    incumbent_recipe: ResearchRecipe | None,
    incumbent_record: dict[str, Any] | None,
) -> str | None:
    if incumbent_recipe is None or incumbent_record is None:
        return None
    if recipe.stage != "exploit" or recipe.parent_recipe != incumbent_recipe.name:
        return None
    if recipe.config.image_backbone != incumbent_recipe.config.image_backbone:
        return None
    if recipe.config.model_dim != incumbent_recipe.config.model_dim:
        return None
    if recipe.use_teacher_provider != incumbent_recipe.use_teacher_provider:
        return None
    checkpoint_path = incumbent_record.get("checkpoint_path")
    return str(checkpoint_path) if checkpoint_path is not None else None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True, default=str) for row in rows)
    path.write_text(f"{payload}\n" if payload else "")


def _write_results_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_TSV_HEADER, delimiter="\t")
        writer.writeheader()
        for row in rows:
            evaluation = row.get("evaluation", {})
            assert isinstance(evaluation, dict)
            benchmark = row.get("benchmark", {})
            assert isinstance(benchmark, dict)
            source_mix = row.get("source_mix", {})
            assert isinstance(source_mix, dict)
            writer.writerow(
                {
                    "run_id": row.get("run_id"),
                    "recipe": row.get("recipe"),
                    "stage": row.get("stage"),
                    "parent_recipe": row.get("parent_recipe") or "",
                    "teacher_seed_mode": row.get("teacher_seed_mode") or "",
                    "teacher_provider": row.get("teacher_provider") or "",
                    "status": row.get("status"),
                    "interim_decision": row.get("interim_decision") or "",
                    "final_decision": row.get("final_decision") or "",
                    "best_so_far": row.get("best_so_far"),
                    "nds": evaluation.get("nd_score", ""),
                    "map": evaluation.get("mean_ap", ""),
                    "val_total": row.get("val", {}).get("total", ""),
                    "latency_ms": benchmark.get("mean_ms", ""),
                    "lidar_share": source_mix.get("lidar", ""),
                    "proposal_share": source_mix.get("proposal", ""),
                    "global_share": source_mix.get("global", ""),
                    "hypothesis": row.get("hypothesis"),
                    "mutation_reason": row.get("mutation_reason"),
                    "decision_reason": row.get("decision_reason") or "",
                    "checkpoint_path": row.get("checkpoint_path") or "",
                }
            )


def _flush_progress_ledgers(artifact_root: Path, records: list[dict[str, Any]]) -> None:
    _write_jsonl(artifact_root / "results.jsonl", records)
    _write_results_tsv(artifact_root / "results.tsv", records)


def _metric_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _metric_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _current_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_worktree_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return bool(result.stdout.strip())


def _canonical_command(
    *,
    dataroot: Path,
    artifact_dir: Path,
    device: str | None,
    max_experiments: int,
) -> str:
    command = [
        "uv",
        "run",
        "tsqbev",
        "research-loop",
        "--dataset-root",
        str(dataroot),
        "--artifact-dir",
        str(artifact_dir.parent),
        "--max-experiments",
        str(max_experiments),
    ]
    if device is not None:
        command.extend(["--device", device])
    return " ".join(command)


def _environment_manifest(device: str | None) -> dict[str, Any]:
    resolved_device = resolve_device(device)
    cuda_device_name = None
    if resolved_device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(resolved_device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_type": resolved_device.type,
        "cuda_device_name": cuda_device_name,
    }


def _write_run_manifest(
    run_dir: Path,
    recipe: ResearchRecipe,
    *,
    dataroot: Path,
    artifact_root: Path,
    device: str | None,
    max_experiments: int,
    teacher_provider_config: TeacherProviderConfig | None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_sha": _current_git_sha(),
        "git_dirty": _git_worktree_dirty(),
        "dataset_root": str(dataroot),
        "artifact_root": str(artifact_root),
        "command": _canonical_command(
            dataroot=dataroot,
            artifact_dir=artifact_root,
            device=device,
            max_experiments=max_experiments,
        ),
        "recipe": _serialize_recipe(recipe),
        "environment": _environment_manifest(device),
        "teacher_provider_config": (
            {
                "kind": teacher_provider_config.kind,
                "cache_dir": teacher_provider_config.cache_dir,
                "checkpoint_path": teacher_provider_config.checkpoint_path,
            }
            if teacher_provider_config is not None
            else None
        ),
    }
    if extra is not None:
        payload["run_record"] = extra
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(payload, indent=2, default=str))


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
) -> dict[str, Any]:
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
    per_batch: list[dict[str, float]] = []
    for batch_index, (batch, _metadata) in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_batch(batch, resolved_device)
        outputs = model(batch)
        seed_bank = outputs["seed_bank"]
        source_ids = seed_bank.source_ids.detach().cpu()
        batch_total = float(source_ids.numel())
        batch_counts = [float((source_ids == source_id).sum()) for source_id in range(3)]
        for source_id, source_count in enumerate(batch_counts):
            counts[source_id] += source_count
        total_queries += batch_total
        if batch_total > 0.0:
            per_batch.append(
                {
                    "lidar": batch_counts[0] / batch_total,
                    "proposal": batch_counts[1] / batch_total,
                    "global": batch_counts[2] / batch_total,
                }
            )
    if total_queries <= 0.0:
        average = {"lidar": 0.0, "proposal": 0.0, "global": 0.0}
    else:
        average = {
            "lidar": float(counts[0] / total_queries),
            "proposal": float(counts[1] / total_queries),
            "global": float(counts[2] / total_queries),
        }
    return {
        "average": average,
        "per_batch": per_batch,
        "batches_measured": len(per_batch),
    }


def _select_better_record(
    current_best: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> tuple[bool, str]:
    candidate_eval = candidate.get("evaluation", {})
    assert isinstance(candidate_eval, dict)
    candidate_nds = float(candidate_eval.get("nd_score", float("-inf")))
    candidate_map = float(candidate_eval.get("mean_ap", float("-inf")))
    candidate_val = float(candidate.get("val", {}).get("total", float("inf")))
    if current_best is None:
        return True, "establishes the first completed baseline for this invocation"
    current_eval = current_best.get("evaluation", {})
    assert isinstance(current_eval, dict)
    current_nds = float(current_eval.get("nd_score", float("-inf")))
    current_map = float(current_eval.get("mean_ap", float("-inf")))
    current_val = float(current_best.get("val", {}).get("total", float("inf")))
    if candidate_nds > current_nds:
        return True, f"improves official mini_val NDS from {current_nds:.4f} to {candidate_nds:.4f}"
    if candidate_nds < current_nds:
        return (
            False,
            f"regresses official mini_val NDS from {current_nds:.4f} "
            f"to {candidate_nds:.4f}",
        )
    if candidate_map > current_map:
        return (
            True,
            f"ties NDS but improves official mini_val mAP from {current_map:.4f} "
            f"to {candidate_map:.4f}",
        )
    if candidate_map < current_map:
        return (
            False,
            f"ties NDS but regresses official mini_val mAP from {current_map:.4f} "
            f"to {candidate_map:.4f}",
        )
    if candidate_val < current_val:
        return (
            True,
            f"ties official metrics and lowers validation total from {current_val:.4f} "
            f"to {candidate_val:.4f}",
        )
    if candidate_val > current_val:
        return (
            False,
            f"ties official metrics and raises validation total from {current_val:.4f} "
            f"to {candidate_val:.4f}",
        )
    return False, "ties the incumbent on official metrics and validation loss"


def _record_rank_key(record: dict[str, Any]) -> tuple[float, float, float]:
    evaluation = record.get("evaluation", {})
    assert isinstance(evaluation, dict)
    return (
        -float(evaluation.get("nd_score", float("-inf"))),
        -float(evaluation.get("mean_ap", float("-inf"))),
        float(record.get("val", {}).get("total", float("inf"))),
    )


def _apply_final_decisions(
    records: list[dict[str, Any]],
    promoted_run_id: int | None,
) -> list[dict[str, Any]]:
    completed = [record for record in records if record.get("status") == "completed"]
    ranked = sorted(completed, key=_record_rank_key)
    ranks = {int(record["run_id"]): rank for rank, record in enumerate(ranked, start=1)}
    for record in records:
        run_id = int(record["run_id"])
        record["final_rank"] = ranks.get(run_id)
        if run_id == promoted_run_id:
            record["final_decision"] = "promote"
        elif record.get("status") == "error":
            record["final_decision"] = "crash"
        else:
            record["final_decision"] = "discard"
    return ranked


def _count_nonzero_classes(label_aps: dict[str, Any]) -> int:
    count = 0
    for distance_map in label_aps.values():
        if not isinstance(distance_map, dict):
            continue
        if any(float(value) > 0.0 for value in distance_map.values()):
            count += 1
    return count


def _teacher_lift(records: list[dict[str, Any]]) -> dict[str, Any]:
    base_records = [
        record
        for record in records
        if record.get("status") == "completed" and not bool(record.get("use_teacher_provider"))
    ]
    teacher_kd_records = [
        record
        for record in records
        if record.get("status") == "completed"
        and bool(record.get("use_teacher_provider"))
        and record.get("teacher_seed_mode") == "off"
    ]
    teacher_seed_records = [
        record
        for record in records
        if record.get("status") == "completed"
        and bool(record.get("use_teacher_provider"))
        and record.get("teacher_seed_mode") == "replace_lidar_refs"
    ]
    if not base_records or (not teacher_kd_records and not teacher_seed_records):
        return {
            "paired": False,
            "passed": False,
            "reason": (
                "no paired teacher-on and teacher-off records were measured in this "
                "invocation"
            ),
        }
    best_base = sorted(base_records, key=_record_rank_key)[0]
    base_eval = best_base.get("evaluation", {})
    assert isinstance(base_eval, dict)
    base_nds = float(base_eval.get("nd_score", 0.0))
    comparisons: dict[str, Any] = {}
    best_lift = float("-inf")
    best_lift_ratio = float("-inf")
    best_teacher_recipe: str | None = None
    for label, pool in (
        ("teacher_kd", teacher_kd_records),
        ("teacher_ref_seed", teacher_seed_records),
    ):
        if not pool:
            continue
        best_teacher = sorted(pool, key=_record_rank_key)[0]
        teacher_eval = best_teacher.get("evaluation", {})
        assert isinstance(teacher_eval, dict)
        teacher_nds = float(teacher_eval.get("nd_score", 0.0))
        abs_lift = teacher_nds - base_nds
        rel_lift = teacher_nds / base_nds if base_nds > 0.0 else float("inf")
        comparisons[label] = {
            "recipe": best_teacher.get("recipe"),
            "nds": teacher_nds,
            "absolute_lift_nds": abs_lift,
            "relative_lift_nds": rel_lift,
        }
        if abs_lift > best_lift:
            best_lift = abs_lift
            best_lift_ratio = rel_lift
            best_teacher_recipe = str(best_teacher.get("recipe"))
    passed = best_lift >= 0.02 or best_lift_ratio >= 2.0
    return {
        "paired": True,
        "passed": passed,
        "baseline_recipe": best_base.get("recipe"),
        "teacher_recipe": best_teacher_recipe,
        "baseline_nds": base_nds,
        "comparisons": comparisons,
        "reason": (
            "teacher lift met the scale-gate threshold"
            if passed
            else "teacher lift did not reach +0.02 NDS or 2x relative NDS"
        ),
    }


def _scale_gate_verdict(
    promoted_record: dict[str, Any] | None,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if promoted_record is None:
        return {
            "authorized": False,
            "reason": "no completed promoted record exists",
            "gates": {},
        }
    evaluation = promoted_record.get("evaluation", {})
    assert isinstance(evaluation, dict)
    source_mix = promoted_record.get("source_mix", {})
    assert isinstance(source_mix, dict)
    diagnostics = promoted_record.get("source_mix_diagnostics", {})
    assert isinstance(diagnostics, dict)
    per_batch = diagnostics.get("per_batch", [])
    assert isinstance(per_batch, list)
    label_aps = evaluation.get("label_aps", {})
    assert isinstance(label_aps, dict)
    tp_errors = evaluation.get("tp_errors", {})
    assert isinstance(tp_errors, dict)
    benchmark = promoted_record.get("benchmark", {})
    assert isinstance(benchmark, dict)

    source_mix_pass = bool(per_batch) and all(
        float(batch.get("lidar", 0.0)) >= 0.2
        and float(batch.get("proposal", 0.0)) >= 0.2
        and max(
            float(batch.get("lidar", 0.0)),
            float(batch.get("proposal", 0.0)),
            float(batch.get("global", 0.0)),
        )
        <= 0.8
        for batch in per_batch
    )
    mini_nds = float(evaluation.get("nd_score", 0.0))
    mini_map = float(evaluation.get("mean_ap", 0.0))
    nonzero_classes = _count_nonzero_classes(label_aps)
    car_distance_aps = label_aps.get("car", {})
    if not isinstance(car_distance_aps, dict):
        car_distance_aps = {}
    car_ap_4m = float(car_distance_aps.get("4.0", 0.0))
    trans_err = tp_errors.get("trans_err")
    translation_gate_pass = isinstance(trans_err, int | float) and float(trans_err) < 1.0
    teacher_lift = _teacher_lift(records)
    gates = {
        "repo_integrity": {
            "passed": False,
            "reason": (
                "ruff/mypy/pytest/export validation are repo-level checks and are "
                "not re-run inside the bounded research invocation"
            ),
        },
        "source_mix_stability": {
            "passed": source_mix_pass and int(diagnostics.get("batches_measured", 0)) >= 8,
            "reason": (
                "source mix remained multimodal across the monitored validation batches"
                if source_mix_pass and int(diagnostics.get("batches_measured", 0)) >= 8
                else "source mix did not yet satisfy the eight-batch multimodality gate"
            ),
            "average": source_mix,
            "batches_measured": int(diagnostics.get("batches_measured", 0)),
        },
        "small_subset_overfit": {
            "passed": False,
            "reason": "the 32-sample overfit protocol has not been run in this invocation",
        },
        "mini_generalization": {
            "passed": (
                mini_nds >= 0.05
                and mini_map >= 0.01
                and nonzero_classes >= 3
                and car_ap_4m >= 0.05
                and translation_gate_pass
            ),
            "reason": "official mini metrics met the promotion threshold"
            if (
                mini_nds >= 0.05
                and mini_map >= 0.01
                and nonzero_classes >= 3
                and car_ap_4m >= 0.05
                and translation_gate_pass
            )
            else "official mini metrics are still below the promotion threshold",
            "nds": mini_nds,
            "map": mini_map,
            "nonzero_classes": nonzero_classes,
            "car_ap_4m": car_ap_4m,
            "translation_error": trans_err,
        },
        "teacher_lift": teacher_lift,
        "efficiency_discipline": {
            "passed": float(benchmark.get("mean_ms", float("inf"))) <= 25.0,
            "reason": (
                "synthetic RTX 5000 latency stayed within the research gate"
                if float(benchmark.get("mean_ms", float("inf"))) <= 25.0
                else "synthetic RTX 5000 latency exceeded the research gate"
            ),
            "mean_ms": float(benchmark.get("mean_ms", float("inf"))),
        },
        "repeatability": {
            "passed": False,
            "reason": "the promoted recipe has not yet been rerun twice under the same conditions",
        },
    }
    gate_statuses: list[bool] = []
    for gate in gates.values():
        assert isinstance(gate, dict)
        gate_statuses.append(bool(gate.get("passed")))
    authorized = all(gate_statuses)
    return {
        "authorized": authorized,
        "reason": (
            "all scale gates passed"
            if authorized
            else "at least one scale gate remains unmet; do not spend 10x compute yet"
        ),
        "gates": gates,
    }


def _recommended_next_steps(
    scale_verdict: dict[str, Any],
    promoted_record: dict[str, Any] | None,
    teacher_provider_config: TeacherProviderConfig | None,
) -> list[str]:
    if promoted_record is None:
        return ["fix the bounded loop until at least one recipe completes cleanly"]
    evaluation = promoted_record.get("evaluation", {})
    assert isinstance(evaluation, dict)
    nds = float(evaluation.get("nd_score", 0.0))
    recommendations: list[str] = []
    if not scale_verdict.get("authorized", False):
        recommendations.append("run the 32-sample overfit gate before any larger-scale training")
    if teacher_provider_config is None:
        recommendations.append(
            "generate a cached external CenterPoint-PointPillar teacher and rerun "
            "a paired teacher-on versus teacher-off mini invocation"
        )
    if nds < 0.05:
        recommendations.append(
            "continue the staged mini loop around the current incumbent instead of "
            "moving to trainval"
        )
    if float(promoted_record.get("benchmark", {}).get("mean_ms", float("inf"))) <= 25.0:
        recommendations.append("preserve the current latency envelope while chasing geometry gains")
    return recommendations


def _leaderboard(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [record for record in records if record.get("status") == "completed"]
    ranked = sorted(completed, key=_record_rank_key)
    return [
        {
            "final_rank": index,
            "recipe": record.get("recipe"),
            "stage": record.get("stage"),
            "teacher_seed_mode": record.get("teacher_seed_mode"),
            "nds": record.get("evaluation", {}).get("nd_score"),
            "map": record.get("evaluation", {}).get("mean_ap"),
            "val_total": record.get("val", {}).get("total"),
            "latency_ms": record.get("benchmark", {}).get("mean_ms"),
        }
        for index, record in enumerate(ranked, start=1)
    ]


def run_bounded_research_loop(
    dataroot: str | Path,
    artifact_dir: str | Path,
    *,
    device: str | None = None,
    max_experiments: int = 5,
    teacher_provider_config: TeacherProviderConfig | None = None,
) -> dict[str, Any]:
    """Run a bounded mini-nuScenes experiment sweep and promote one incumbent."""

    ensure_research_loop_enabled()
    root = Path(dataroot)
    artifact_root = Path(artifact_dir) / "research_loop"
    artifact_root.mkdir(parents=True, exist_ok=True)
    max_experiments = max(1, max_experiments)

    records: list[dict[str, Any]] = []
    incumbent_record: dict[str, Any] | None = None
    incumbent_recipe: ResearchRecipe | None = None
    candidate_queue = _initial_recipes(
        artifact_root,
        teacher_provider_available=teacher_provider_config is not None,
    )[:max_experiments]
    initial_recipe_count = len(candidate_queue)
    recipe_index = 0

    while recipe_index < len(candidate_queue) and recipe_index < max_experiments:
        recipe = candidate_queue[recipe_index]
        run_id = recipe_index + 1
        run_dir = artifact_root / recipe.name
        run_teacher_provider_config = (
            teacher_provider_config if recipe.use_teacher_provider else None
        )
        tracker = start_experiment_tracking(
            artifact_dir=run_dir,
            config=recipe.config,
            metadata=TrackingMetadata(
                suite="research",
                dataset="nuscenes",
                job_type="research-loop",
                run_name=recipe.name,
                group=artifact_root.name,
                tags=(
                    "research-loop",
                    recipe.stage,
                    "v1.0-mini",
                    recipe.config.image_backbone,
                    recipe.config.teacher_seed_mode,
                    "teacher-on" if recipe.use_teacher_provider else "teacher-off",
                ),
                extra_config={
                    "run_id": run_id,
                    "parent_recipe": recipe.parent_recipe,
                    "max_experiments": max_experiments,
                    "use_teacher_provider": recipe.use_teacher_provider,
                    "teacher_provider": (
                        teacher_provider_config.kind
                        if teacher_provider_config is not None and recipe.use_teacher_provider
                        else None
                    ),
                    "recipe": _serialize_recipe(recipe),
                },
            ),
            config_payload={
                "model": recipe.config.model_dump(),
                "train": {
                    "epochs": recipe.epochs,
                    "max_train_steps": recipe.max_train_steps,
                    "lr": recipe.lr,
                    "grad_accum_steps": recipe.grad_accum_steps,
                    "batch_size": recipe.batch_size,
                    "num_workers": recipe.num_workers,
                },
            },
        )
        print(
            "[research] "
            f"starting run_id={run_id} recipe={recipe.name} stage={recipe.stage} "
            f"teacher_seed_mode={recipe.config.teacher_seed_mode} "
            f"use_teacher_provider={recipe.use_teacher_provider} "
            f"parent={recipe.parent_recipe or '-'}"
        )
        record: dict[str, Any] = {
            "run_id": run_id,
            **_serialize_recipe(recipe),
            "status": "started",
            "teacher_seed_mode": recipe.config.teacher_seed_mode,
            "teacher_provider": (
                teacher_provider_config.kind
                if teacher_provider_config is not None and recipe.use_teacher_provider
                else None
            ),
            "interim_decision": "pending",
            "final_decision": "pending",
            "best_so_far": False,
        }
        _write_run_manifest(
            run_dir,
            recipe,
            dataroot=root,
            artifact_root=artifact_root,
            device=device,
            max_experiments=max_experiments,
            teacher_provider_config=run_teacher_provider_config,
            extra={"run_id": run_id, "status": "started"},
        )
        try:
            durations_s: dict[str, float] = {}

            start_time = time.perf_counter()
            train_result = fit_nuscenes(
                dataroot=root,
                artifact_dir=run_dir,
                config=recipe.config,
                version="v1.0-mini",
                train_split="mini_train",
                val_split="mini_val",
                epochs=recipe.epochs,
                max_train_steps=recipe.max_train_steps,
                lr=recipe.lr,
                grad_accum_steps=recipe.grad_accum_steps,
                batch_size=recipe.batch_size,
                num_workers=recipe.num_workers,
                device=device,
                teacher_provider_config=run_teacher_provider_config,
                init_checkpoint=_warm_start_checkpoint_for_recipe(
                    recipe,
                    incumbent_recipe,
                    incumbent_record,
                ),
                use_amp=False,
                log_every_steps=25,
                tracker=tracker,
            )
            durations_s["train"] = time.perf_counter() - start_time

            start_time = time.perf_counter()
            bench = benchmark_forward(
                recipe.config,
                steps=10,
                warmup=3,
                batch_size=1,
                device=device,
                image_height=256,
                image_width=704,
            )
            durations_s["benchmark"] = time.perf_counter() - start_time

            checkpoint_path = Path(str(train_result["checkpoint_path"]))
            model, _ = load_model_from_checkpoint(checkpoint_path)

            prediction_path = run_dir / "mini_predictions.json"
            start_time = time.perf_counter()
            export_nuscenes_predictions(
                model=model,
                dataroot=root,
                version="v1.0-mini",
                split="mini_val",
                output_path=prediction_path,
                score_threshold=recipe.score_threshold,
                top_k=recipe.top_k,
                device=device,
                teacher_provider_config=run_teacher_provider_config,
            )
            durations_s["export"] = time.perf_counter() - start_time

            start_time = time.perf_counter()
            evaluation = evaluate_nuscenes_predictions(
                dataroot=root,
                version="v1.0-mini",
                split="mini_val",
                result_path=prediction_path,
                output_dir=run_dir / "mini_eval",
            )
            durations_s["evaluate"] = time.perf_counter() - start_time

            start_time = time.perf_counter()
            source_mix_diagnostics = _measure_source_mix(
                model,
                root,
                version="v1.0-mini",
                split="mini_val",
                device=device,
                teacher_provider_config=run_teacher_provider_config,
            )
            durations_s["source_mix"] = time.perf_counter() - start_time
            train_samples = train_result["train_samples"]
            val_samples = train_result["val_samples"]
            assert isinstance(train_samples, int)
            assert isinstance(val_samples, int)

            record.update(
                {
                    "status": "completed",
                    "train": train_result["last_train"],
                    "val": train_result["last_val"],
                    "benchmark": bench,
                    "checkpoint_path": str(checkpoint_path),
                    "prediction_path": str(prediction_path),
                    "evaluation": evaluation,
                    "source_mix": source_mix_diagnostics["average"],
                    "source_mix_diagnostics": source_mix_diagnostics,
                    "durations_s": durations_s,
                    "train_samples": train_samples,
                    "val_samples": val_samples,
                }
            )
            better, reason = _select_better_record(incumbent_record, record)
            record["interim_decision"] = "advance" if better else "reject"
            record["decision_reason"] = reason
            record["best_so_far"] = bool(better)
            if better:
                incumbent_record = record
                incumbent_recipe = recipe
            epochs_run = _metric_int(train_result.get("epochs", 0))
            average_mix = cast(dict[str, object], source_mix_diagnostics["average"])
            tracker.log(
                {
                    "epoch": epochs_run,
                    "eval_nds": _metric_float(evaluation.get("nd_score", 0.0)),
                    "eval_map": _metric_float(evaluation.get("mean_ap", 0.0)),
                    "benchmark_mean_ms": _metric_float(bench.get("mean_ms", 0.0)),
                    "benchmark_p95_ms": _metric_float(bench.get("p95_ms", 0.0)),
                    "source_mix_lidar": _metric_float(average_mix.get("lidar", 0.0)),
                    "source_mix_proposal": _metric_float(average_mix.get("proposal", 0.0)),
                    "source_mix_global": _metric_float(average_mix.get("global", 0.0)),
                },
                step=epochs_run,
            )
        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "error": repr(exc),
                    "interim_decision": "crash",
                    "decision_reason": "runtime error during bounded research invocation",
                }
            )
            tracker.summary({"error": repr(exc)})
        records.append(record)
        _write_run_manifest(
            run_dir,
            recipe,
            dataroot=root,
            artifact_root=artifact_root,
            device=device,
            max_experiments=max_experiments,
            teacher_provider_config=run_teacher_provider_config,
            extra=record,
        )
        _flush_progress_ledgers(artifact_root, records)
        evaluation = record.get("evaluation", {})
        assert isinstance(evaluation, dict)
        val_metrics = record.get("val", {})
        assert isinstance(val_metrics, dict)
        tracker.summary(
            {
                "run_id": run_id,
                "recipe": recipe.name,
                "stage": recipe.stage,
                "status": record.get("status"),
                "interim_decision": record.get("interim_decision"),
                "decision_reason": record.get("decision_reason"),
                "checkpoint_path": record.get("checkpoint_path"),
                "prediction_path": record.get("prediction_path"),
                "eval_nds": _metric_float(evaluation.get("nd_score", 0.0)),
                "eval_map": _metric_float(evaluation.get("mean_ap", 0.0)),
                "val_total": _metric_float(val_metrics.get("total", 0.0)),
            }
        )
        print(
            "[research] "
            f"finished run_id={run_id} recipe={recipe.name} status={record.get('status')} "
            f"decision={record.get('interim_decision')} "
            f"nds={_metric_float(evaluation.get('nd_score', 0.0)):.6f} "
            f"map={_metric_float(evaluation.get('mean_ap', 0.0)):.6f} "
            f"val_total={_metric_float(val_metrics.get('total', 0.0)):.4f}"
        )
        tracker.finish(status="completed" if record.get("status") == "completed" else "failed")

        if (
            recipe_index + 1 == initial_recipe_count
            and incumbent_record is not None
            and incumbent_recipe is not None
        ):
            remaining_budget = max_experiments - len(candidate_queue)
            candidate_queue.extend(
                _build_exploitation_recipes(
                    incumbent_recipe,
                    incumbent_record,
                    teacher_provider_config,
                    remaining_budget,
                )
            )
        recipe_index += 1

    promoted_run_id = int(incumbent_record["run_id"]) if incumbent_record is not None else None
    ranked = _apply_final_decisions(records, promoted_run_id)
    _flush_progress_ledgers(artifact_root, records)

    if incumbent_record is None or incumbent_recipe is None:
        failed_summary: dict[str, Any] = {
            "status": "failed",
            "records_path": str(artifact_root / "results.jsonl"),
            "results_tsv_path": str(artifact_root / "results.tsv"),
            "records": records,
            "scale_gate_verdict": {
                "authorized": False,
                "reason": "no completed promoted record exists",
                "gates": {},
            },
        }
        (artifact_root / "summary.json").write_text(
            json.dumps(failed_summary, indent=2, default=str)
        )
        return failed_summary

    scale_verdict = _scale_gate_verdict(incumbent_record, records)
    summary: dict[str, Any] = {
        "status": "completed",
        "reference_workflow": "karpathy/autoresearch",
        "selected_recipe": incumbent_recipe.name,
        "records_path": str(artifact_root / "results.jsonl"),
        "results_tsv_path": str(artifact_root / "results.tsv"),
        "selected_checkpoint": incumbent_record["checkpoint_path"],
        "selected_record": incumbent_record,
        "evaluation": incumbent_record["evaluation"],
        "leaderboard": _leaderboard(records),
        "scale_gate_verdict": scale_verdict,
        "recommended_next_steps": _recommended_next_steps(
            scale_verdict,
            incumbent_record,
            teacher_provider_config,
        ),
        "recipes": [_serialize_recipe(recipe) for recipe in candidate_queue[:max_experiments]],
        "ranked_recipe_names": [str(record.get("recipe")) for record in ranked],
    }
    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
