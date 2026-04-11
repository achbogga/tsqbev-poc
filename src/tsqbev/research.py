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

from tsqbev.bevdet_env import (
    CONTROL_BEVDET_CONFIG,
    DEFAULT_BEVDET_IMAGE_TAG,
    PRIMARY_BEVDET_CONFIG,
    run_bevdet_public_student,
)
from tsqbev.checkpoints import load_model_from_checkpoint
from tsqbev.config import ModelConfig
from tsqbev.datasets import NuScenesDataset, collate_scene_examples
from tsqbev.eval_nuscenes import (
    export_and_evaluate_nuscenes_grid,
    prediction_geometry_diagnostics,
)
from tsqbev.research_guard import ensure_research_loop_enabled
from tsqbev.research_memory import safe_build_research_brief, safe_sync_research_memory
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
    "selected_epoch",
    "best_epoch",
    "hypothesis",
    "mutation_reason",
    "root_cause_verdict",
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
    weight_decay: float = 1e-4
    epochs: int = 6
    max_train_steps: int | None = 960
    num_workers: int = 4
    score_threshold: float = 0.05
    top_k: int = 112
    init_checkpoint: str | None = None
    optimizer_schedule: Literal["cosine", "constant"] = "cosine"
    grad_clip_norm: float | None = 1.0
    keep_best_checkpoint: bool = True
    augmentation_mode: Literal["off", "moderate", "strong"] = "off"
    enable_teacher_distillation: bool = True
    loss_mode: Literal["baseline", "focal_hardneg", "quality_focal"] = "baseline"
    hard_negative_ratio: int = 3
    hard_negative_cap: int = 96
    teacher_anchor_quality_class_weight: float = 0.0
    teacher_region_objectness_weight: float = 0.0
    teacher_region_class_weight: float = 0.0
    teacher_region_radius_m: float = 4.0
    score_threshold_candidates: tuple[float, ...] = (0.05,)
    top_k_candidates: tuple[int, ...] = (112,)
    official_eval_every_epochs: int | None = None
    official_eval_score_threshold: float = 0.20
    official_eval_top_k: int = 40
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0
    early_stop_min_epochs: int = 0
    skip_training: bool = False
    execution_backend: Literal["tsqbev_local", "bevdet_official"] = "tsqbev_local"
    external_repo_root: str | None = None
    external_config_relpath: str | None = None
    external_image_tag: str | None = None
    external_version: str | None = None
    comparison_tier: Literal["frontier_comparable", "bounded_local", "smoke_control"] = (
        "frontier_comparable"
    )
    promotion_eligible: bool = True
    frontier_claim_eligible: bool = True
    exact_public_contract: bool = False


@dataclass(slots=True)
class ResearchProposal:
    """Supervisor-selected bounded research thesis for one invocation."""

    proposal_id: str
    objective: str
    targeted_bottleneck: str | None
    launch_tags: list[str]
    exploitation_tags: list[str]
    suppress_tags: list[str]
    rationale: list[str]
    kill_conditions: list[str]
    force_tags_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "objective": self.objective,
            "targeted_bottleneck": self.targeted_bottleneck,
            "launch_tags": list(self.launch_tags),
            "exploitation_tags": list(self.exploitation_tags),
            "suppress_tags": list(self.suppress_tags),
            "rationale": list(self.rationale),
            "kill_conditions": list(self.kill_conditions),
            "force_tags_only": self.force_tags_only,
        }


def _normalize_proposal(
    proposal: ResearchProposal | dict[str, Any] | None,
    *,
    fallback_policy: dict[str, Any] | None = None,
) -> ResearchProposal | None:
    if proposal is None and fallback_policy is None:
        return None
    if isinstance(proposal, ResearchProposal):
        base = proposal.to_dict()
    elif isinstance(proposal, dict):
        base = dict(proposal)
    else:
        base = {}
    policy = fallback_policy or {}
    launch_tags = [str(tag) for tag in base.get("launch_tags", [])]
    exploitation_tags = [str(tag) for tag in base.get("exploitation_tags", [])]
    if not launch_tags:
        launch_tags = [str(tag) for tag in policy.get("priority_tags", [])]
    if not exploitation_tags:
        exploitation_tags = launch_tags.copy()
    suppress_tags = [str(tag) for tag in base.get("suppress_tags", policy.get("suppress_tags", []))]
    rationale = [str(item) for item in base.get("rationale", policy.get("reasons", []))]
    kill_conditions = [str(item) for item in base.get("kill_conditions", [])]
    objective = str(base.get("objective", "improve official mini-val frontier"))
    targeted_bottleneck = base.get("targeted_bottleneck")
    default_proposal_id = f"proposal-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    proposal_id = str(base.get("proposal_id", default_proposal_id))
    force_tags_only = bool(base.get("force_tags_only", policy.get("force_priority_only", False)))
    return ResearchProposal(
        proposal_id=proposal_id,
        objective=objective,
        targeted_bottleneck=(
            None if targeted_bottleneck is None else str(targeted_bottleneck)
        ),
        launch_tags=launch_tags,
        exploitation_tags=exploitation_tags,
        suppress_tags=suppress_tags,
        rationale=rationale,
        kill_conditions=kill_conditions,
        force_tags_only=force_tags_only,
    )


def _proposal_policy(
    proposal: ResearchProposal | None,
    *,
    phase: Literal["launch", "exploit"],
) -> dict[str, Any]:
    if proposal is None:
        return {}
    priority_tags = proposal.launch_tags if phase == "launch" else proposal.exploitation_tags
    return {
        "priority_tags": list(priority_tags),
        "suppress_tags": list(proposal.suppress_tags),
        "force_priority_only": bool(proposal.force_tags_only),
        "reasons": list(proposal.rationale),
    }


_FRONTIER_LAUNCH_TAGS = {
    "dino_v3",
    "dino_v3_bridge",
    "bevformer_v2_perspective_supervision",
    "teacher_bootstrap",
    "lightweight_bridge",
    "gated_cross_attention",
    "teacher_side_foundation",
    "shared_world_latent",
    "world_aligned_distillation",
    "world_latent_distillation",
    "bevfusion_teacher",
    "openpcdet_teacher",
    "sparse4d_efficiency",
    "sam21_offline_support",
}

_PUBLIC_STUDENT_LAUNCH_TAGS = {
    "public_student_replacement",
    "bevdet_public_student",
    "bevdepth_temporal_student",
    "official_box_coder",
    "camera_bev_working_baseline",
    "exact_public_reproduction",
    "mini_smoke_only",
    "artifact_persistence_required",
    "noncomparable_runs_do_not_count",
}


def _priority_tags_request_frontier(priority_tags: list[str]) -> bool:
    return any(tag in _FRONTIER_LAUNCH_TAGS for tag in priority_tags)


def _priority_tags_request_public_student(priority_tags: list[str]) -> bool:
    return any(tag in _PUBLIC_STUDENT_LAUNCH_TAGS for tag in priority_tags)


def _is_frontier_recipe(recipe: ResearchRecipe) -> bool:
    return recipe.execution_backend == "bevdet_official" or (
        recipe.config.image_backbone.startswith("dinov3_")
        or recipe.config.sam2_region_prior_mode != "off"
        or recipe.config.fusion_style == "gated_latent_cross_attn"
    )


def _public_student_base_config() -> ModelConfig:
    return ModelConfig.rtx5000_nuscenes_baseline().model_copy(
        update={"freeze_image_backbone": True}
    )


def _public_bevdet_primary_recipe() -> ResearchRecipe:
    return ResearchRecipe(
        name="public_bevdet_r50_cbgs",
        note=(
            "full public-student replacement using the official BEVDet R50-CBGS deployment "
            "baseline with its native box coder, decoder, and postprocess"
        ),
        hypothesis=(
            "a public working student with a proven detector head/decoder contract should beat "
            "the custom TSQBEV student on correctness immediately, even before teacher tricks"
        ),
        mutation_reason=(
            "replace the custom student family with the official BEVDet deployment baseline "
            "because the TSQBEV head/decoder contract is fundamentally broken"
        ),
        config=_public_student_base_config(),
        stage="baseline",
        use_teacher_provider=False,
        batch_size=1,
        grad_accum_steps=1,
        lr=2e-4,
        weight_decay=1e-2,
        epochs=4,
        max_train_steps=None,
        num_workers=2,
        score_threshold=0.10,
        top_k=500,
        keep_best_checkpoint=True,
        enable_teacher_distillation=False,
        loss_mode="baseline",
        official_eval_every_epochs=1,
        official_eval_score_threshold=0.10,
        official_eval_top_k=500,
        early_stop_patience=None,
        execution_backend="bevdet_official",
        external_repo_root="/home/achbogga/projects/BEVDet",
        external_config_relpath=PRIMARY_BEVDET_CONFIG,
        external_image_tag=DEFAULT_BEVDET_IMAGE_TAG,
        external_version="v1.0-mini",
        comparison_tier="smoke_control",
        promotion_eligible=False,
        frontier_claim_eligible=False,
        exact_public_contract=False,
    )


def _make_public_bevdet_temporal_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name="public_bevdet_r50_4d_depth_cbgs",
        note=(
            "official BEVDet temporal-depth upgrade once the plain deployment baseline proves "
            "the public student contract"
        ),
        hypothesis=(
            "once the public baseline is stable, temporal depth lifting is the cleanest official "
            "next step with a stronger published NDS than the plain BEVDet deployment baseline"
        ),
        mutation_reason=(
            "advance from BEVDet R50-CBGS to the official BEVDet R50-4D-Depth-CBGS temporal "
            "student after establishing a nonzero public baseline"
        ),
        epochs=4,
        execution_backend="bevdet_official",
        external_repo_root="/home/achbogga/projects/BEVDet",
        external_config_relpath=CONTROL_BEVDET_CONFIG,
        external_image_tag=DEFAULT_BEVDET_IMAGE_TAG,
        external_version="v1.0-mini",
        comparison_tier="smoke_control",
        promotion_eligible=False,
        frontier_claim_eligible=False,
        exact_public_contract=False,
    )


def _make_public_bevdet_primary_rerun(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name="public_bevdet_r50_cbgs_rerun",
        note="rerun the official BEVDet deployment baseline to verify it is a real control line",
        hypothesis=(
            "a public student replacement should reproduce nonzero official metrics before we "
            "layer on temporal or teacher-side complexity"
        ),
        mutation_reason="repeat the BEVDet baseline under the same mini protocol",
        epochs=4,
        execution_backend="bevdet_official",
        external_repo_root="/home/achbogga/projects/BEVDet",
        external_config_relpath=PRIMARY_BEVDET_CONFIG,
        external_image_tag=DEFAULT_BEVDET_IMAGE_TAG,
        external_version="v1.0-mini",
        comparison_tier="smoke_control",
        promotion_eligible=False,
        frontier_claim_eligible=False,
        exact_public_contract=False,
    )


def _frontier_vits16_config(*, teacher_provider_available: bool) -> ModelConfig:
    config = ModelConfig.rtx5000_nuscenes_dinov3_teacher()
    if teacher_provider_available:
        return config
    return config.model_copy(
        update={
            "teacher_seed_mode": "off",
            "router_mode": "tri_source",
            "ranking_mode": "class_times_objectness",
        }
    )


def _frontier_vitb16_config(*, teacher_provider_available: bool) -> ModelConfig:
    base = _frontier_vits16_config(teacher_provider_available=teacher_provider_available)
    return base.model_copy(
        update={
            "image_backbone": "dinov3_vitb16",
            "foundation_weights": (
                "/home/achbogga/projects/research/dinov3_weights/"
                "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            ),
        }
    )


def _frontier_vits16_recipe(*, teacher_provider_available: bool) -> ResearchRecipe:
    return ResearchRecipe(
        name=(
            "frontier_dinov3_teacher_distill_vits16"
            if teacher_provider_available
            else "frontier_dinov3_camera_only_vits16"
        ),
        note=(
            "hard-pivot executable slice: DINOv3 ViT-S/16 projection, SAM 2.1 region priors, "
            "teacher-guided ranking, and aggressive official-eval gating"
        ),
        hypothesis=(
            "the next real gain should come from stronger camera semantics and tighter "
            "teacher-guided geometry constraints, not another local MobileNet query tweak"
        ),
        mutation_reason=(
            "replace legacy carryover recipes with the DINOv3/SAM2 teacher-distillation "
            "frontier slice and let official metrics decide early"
        ),
        config=_frontier_vits16_config(teacher_provider_available=teacher_provider_available),
        stage="baseline",
        use_teacher_provider=teacher_provider_available,
        batch_size=1,
        grad_accum_steps=4,
        lr=1e-4,
        weight_decay=1e-4,
        epochs=30,
        max_train_steps=None,
        num_workers=4,
        score_threshold=0.20,
        top_k=40,
        optimizer_schedule="cosine",
        grad_clip_norm=1.0,
        keep_best_checkpoint=True,
        augmentation_mode="off",
        enable_teacher_distillation=teacher_provider_available,
        loss_mode="quality_focal",
        teacher_anchor_quality_class_weight=0.45 if teacher_provider_available else 0.0,
        teacher_region_objectness_weight=0.12 if teacher_provider_available else 0.0,
        teacher_region_class_weight=0.12 if teacher_provider_available else 0.0,
        teacher_region_radius_m=6.0,
        score_threshold_candidates=(0.15, 0.20, 0.24, 0.28),
        top_k_candidates=(32, 40, 48),
        official_eval_every_epochs=5,
        official_eval_score_threshold=0.20,
        official_eval_top_k=40,
        early_stop_patience=3,
        early_stop_min_delta=0.02,
        early_stop_min_epochs=10,
    )


def _frontier_light_bridge_recipe(*, teacher_provider_available: bool) -> ResearchRecipe:
    if teacher_provider_available:
        base = ModelConfig.rtx5000_nuscenes_bridge_teacher_bootstrap()
    else:
        base = ModelConfig.rtx5000_nuscenes_bridge_teacher().model_copy(
            update={"teacher_seed_mode": "off"}
        )
    return ResearchRecipe(
        name=(
            "frontier_light_bridge_teacher_bootstrap"
            if teacher_provider_available
            else "frontier_light_bridge_camera_only"
        ),
        note=(
            "embedded-first hard pivot: lightweight student backbone, gated latent camera bridge, "
            "teacher-anchor bootstrap, and official-metric guardrails"
        ),
        hypothesis=(
            "the repeated DINOv3-on-student collapse means foundation models should stay on the "
            "teacher side while the deployable student learns a lighter geometry bridge and uses "
            "teacher-anchor bootstrap to stabilize classes and objectness"
        ),
        mutation_reason=(
            "move frontier foundations to the teacher side, restore teacher-anchor bootstrap, "
            "and replace the student fusion path with a lightweight gated latent bridge under "
            "the same official-eval contract"
        ),
        config=base,
        stage="baseline",
        use_teacher_provider=teacher_provider_available,
        batch_size=2,
        grad_accum_steps=2,
        lr=2e-4,
        weight_decay=1e-4,
        epochs=30,
        max_train_steps=None,
        num_workers=4,
        score_threshold=0.20,
        top_k=40,
        optimizer_schedule="cosine",
        grad_clip_norm=1.0,
        keep_best_checkpoint=True,
        augmentation_mode="off",
        enable_teacher_distillation=teacher_provider_available,
        loss_mode="quality_focal",
        teacher_anchor_quality_class_weight=0.20 if teacher_provider_available else 0.0,
        teacher_region_objectness_weight=0.10 if teacher_provider_available else 0.0,
        teacher_region_class_weight=0.10 if teacher_provider_available else 0.0,
        teacher_region_radius_m=4.0,
        score_threshold_candidates=(0.16, 0.20, 0.24, 0.28),
        top_k_candidates=(32, 40, 48),
        official_eval_every_epochs=5,
        official_eval_score_threshold=0.20,
        official_eval_top_k=40,
        early_stop_patience=2,
        early_stop_min_delta=0.02,
        early_stop_min_epochs=5,
    )


def _make_frontier_light_bridge_effb0_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_effb0",
        note=(
            "scale the lightweight bridge from MobileNetV3 to EfficientNet-B0 without "
            "changing the bridge contract"
        ),
        hypothesis=(
            "if the lightweight bridge is representation-limited, EfficientNet-B0 is the "
            "smallest camera-scale increase that still respects embedded deployment"
        ),
        mutation_reason=(
            "swap to EfficientNet-B0 while preserving lightweight gated bridge and "
            "teacher-only frontier guidance"
        ),
        config=recipe.config.model_copy(
            update={
                "image_backbone": "efficientnet_b0",
                "pretrained_image_backbone": True,
                "freeze_image_backbone": True,
            }
        ),
        stage="explore",
    )


def _make_frontier_vitb16_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_vitb16",
        note=(
            "scale the hard-pivot camera branch to DINOv3 ViT-B/16 with activation "
            "checkpointing still enabled"
        ),
        hypothesis=(
            "the teacher-aligned hard pivot may still be camera-capacity limited; a bounded "
            "move from ViT-S/16 to ViT-B/16 is the cleanest frontier scale-up"
        ),
        mutation_reason=(
            "upgrade the DINOv3 camera backbone to ViT-B/16 without changing the rest of "
            "the launch contract"
        ),
        config=_frontier_vitb16_config(
            teacher_provider_available=recipe.use_teacher_provider,
        ),
        stage="explore",
        batch_size=1,
        grad_accum_steps=max(recipe.grad_accum_steps, 6),
    )


def _make_frontier_world_distill_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_world_distill",
        note=(
            "increase teacher-aligned quality pressure on the DINOv3/SAM2 branch as the "
            "current executable proxy for world-aligned distillation"
        ),
        hypothesis=(
            "if the frontier branch is still losing geometry despite stronger camera features, "
            "the next bounded move is more teacher alignment rather than a legacy query mutation"
        ),
        mutation_reason=(
            "raise teacher quality and region supervision on the hard-pivot branch while "
            "keeping the rest of the recipe fixed"
        ),
        stage="exploit",
        teacher_anchor_quality_class_weight=max(
            recipe.teacher_anchor_quality_class_weight,
            0.55,
        ),
        teacher_region_objectness_weight=max(recipe.teacher_region_objectness_weight, 0.15),
        teacher_region_class_weight=max(recipe.teacher_region_class_weight, 0.15),
        score_threshold_candidates=(0.16, 0.20, 0.24, 0.28),
        top_k_candidates=(32, 40, 48),
    )


def _make_frontier_sam2_ablation_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_no_sam2",
        note=(
            "ablate SAM 2.1 priors while keeping the DINOv3 and teacher-distillation path fixed"
        ),
        hypothesis=(
            "if the frontier branch destabilizes geometry, SAM 2.1 proposal priors need to be "
            "measured explicitly instead of assumed helpful"
        ),
        mutation_reason="disable SAM 2.1 priors as a paired hard-pivot ablation",
        config=recipe.config.model_copy(
            update={
                "sam2_region_prior_mode": "off",
                "sam2_region_prior_weight": 0.0,
            }
        ),
        stage="exploit",
    )


def _make_frontier_teacher_control_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_teacher_control",
        note=(
            "paired hard-pivot control with DINOv3/SAM2 still active but teacher anchoring "
            "and distillation disabled"
        ),
        hypothesis=(
            "the hard-pivot branch must prove that teacher supervision, not just the new camera "
            "backbone, is driving any gain"
        ),
        mutation_reason=(
            "turn teacher seeding and distillation off while preserving the hard-pivot camera "
            "branch and the official-eval launch contract"
        ),
        config=recipe.config.model_copy(
            update={
                "teacher_seed_mode": "off",
                "router_mode": "tri_source",
                "ranking_mode": "class_times_objectness",
            }
        ),
        stage="exploit",
        use_teacher_provider=False,
        enable_teacher_distillation=False,
        teacher_anchor_quality_class_weight=0.0,
        teacher_region_objectness_weight=0.0,
        teacher_region_class_weight=0.0,
    )


def _make_frontier_official_guardrail_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_official_guardrail",
        note="eval-only guardrail sweep around the current frontier export boundary",
        hypothesis=(
            "the hard-pivot branch should be measured on the official operating boundary before "
            "opening another training mutation"
        ),
        mutation_reason=(
            "run an eval-only guardrail sweep on the current frontier checkpoint to separate "
            "training gains from export-boundary failures"
        ),
        stage="exploit",
        score_threshold_candidates=(0.16, 0.20, 0.24, 0.28, 0.32),
        top_k_candidates=(32, 40, 48, 56),
        skip_training=True,
    )


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
        optimizer_schedule=parent.optimizer_schedule,
        grad_clip_norm=parent.grad_clip_norm,
        keep_best_checkpoint=parent.keep_best_checkpoint,
        loss_mode=parent.loss_mode,
        hard_negative_ratio=parent.hard_negative_ratio,
        hard_negative_cap=parent.hard_negative_cap,
        score_threshold_candidates=parent.score_threshold_candidates,
        top_k_candidates=parent.top_k_candidates,
    )


def _updated_config(
    config: ModelConfig,
    *,
    image_backbone: str | None = None,
    pretrained_image_backbone: bool | None = None,
    freeze_image_backbone: bool | None = None,
    router_mode: str | None = None,
    q_lidar: int | None = None,
    q_2d: int | None = None,
    q_global: int | None = None,
    max_object_queries: int | None = None,
    proposals_per_view: int | None = None,
    teacher_seed_mode: str | None = None,
    teacher_seed_selection_mode: str | None = None,
    ranking_mode: str | None = None,
    anchor_first_min_proposal: int | None = None,
    anchor_first_min_global: int | None = None,
) -> ModelConfig:
    updates: dict[str, object] = {}
    pillar_updates: dict[str, object] = {}
    if image_backbone is not None:
        updates["image_backbone"] = image_backbone
    if pretrained_image_backbone is not None:
        updates["pretrained_image_backbone"] = pretrained_image_backbone
    if freeze_image_backbone is not None:
        updates["freeze_image_backbone"] = freeze_image_backbone
    if router_mode is not None:
        updates["router_mode"] = router_mode
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
    if teacher_seed_selection_mode is not None:
        updates["teacher_seed_selection_mode"] = teacher_seed_selection_mode
    if ranking_mode is not None:
        updates["ranking_mode"] = ranking_mode
    if anchor_first_min_proposal is not None:
        updates["anchor_first_min_proposal"] = anchor_first_min_proposal
    if anchor_first_min_global is not None:
        updates["anchor_first_min_global"] = anchor_first_min_global
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
    weight_decay: float | None = None,
    epochs: int | None = None,
    max_train_steps: int | None = None,
    num_workers: int | None = None,
    score_threshold: float | None = None,
    top_k: int | None = None,
    init_checkpoint: str | None = None,
    optimizer_schedule: Literal["cosine", "constant"] | None = None,
    grad_clip_norm: float | None = None,
    keep_best_checkpoint: bool | None = None,
    augmentation_mode: Literal["off", "moderate", "strong"] | None = None,
    enable_teacher_distillation: bool | None = None,
    loss_mode: Literal["baseline", "focal_hardneg", "quality_focal"] | None = None,
    hard_negative_ratio: int | None = None,
    hard_negative_cap: int | None = None,
    teacher_anchor_quality_class_weight: float | None = None,
    teacher_region_objectness_weight: float | None = None,
    teacher_region_class_weight: float | None = None,
    teacher_region_radius_m: float | None = None,
    score_threshold_candidates: tuple[float, ...] | None = None,
    top_k_candidates: tuple[int, ...] | None = None,
    official_eval_every_epochs: int | None = None,
    official_eval_score_threshold: float | None = None,
    official_eval_top_k: int | None = None,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float | None = None,
    early_stop_min_epochs: int | None = None,
    skip_training: bool | None = None,
    execution_backend: Literal["tsqbev_local", "bevdet_official"] | None = None,
    external_repo_root: str | None = None,
    external_config_relpath: str | None = None,
    external_image_tag: str | None = None,
    external_version: str | None = None,
    comparison_tier: Literal["frontier_comparable", "bounded_local", "smoke_control"] | None = None,
    promotion_eligible: bool | None = None,
    frontier_claim_eligible: bool | None = None,
    exact_public_contract: bool | None = None,
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
        weight_decay=recipe.weight_decay if weight_decay is None else weight_decay,
        epochs=recipe.epochs if epochs is None else epochs,
        max_train_steps=recipe.max_train_steps if max_train_steps is None else max_train_steps,
        num_workers=recipe.num_workers if num_workers is None else num_workers,
        score_threshold=recipe.score_threshold if score_threshold is None else score_threshold,
        top_k=recipe.top_k if top_k is None else top_k,
        init_checkpoint=recipe.init_checkpoint if init_checkpoint is None else init_checkpoint,
        optimizer_schedule=(
            recipe.optimizer_schedule if optimizer_schedule is None else optimizer_schedule
        ),
        grad_clip_norm=recipe.grad_clip_norm if grad_clip_norm is None else grad_clip_norm,
        keep_best_checkpoint=(
            recipe.keep_best_checkpoint if keep_best_checkpoint is None else keep_best_checkpoint
        ),
        augmentation_mode=(
            recipe.augmentation_mode if augmentation_mode is None else augmentation_mode
        ),
        enable_teacher_distillation=(
            recipe.enable_teacher_distillation
            if enable_teacher_distillation is None
            else enable_teacher_distillation
        ),
        loss_mode=recipe.loss_mode if loss_mode is None else loss_mode,
        hard_negative_ratio=(
            recipe.hard_negative_ratio if hard_negative_ratio is None else hard_negative_ratio
        ),
        hard_negative_cap=(
            recipe.hard_negative_cap if hard_negative_cap is None else hard_negative_cap
        ),
        teacher_anchor_quality_class_weight=(
            recipe.teacher_anchor_quality_class_weight
            if teacher_anchor_quality_class_weight is None
            else teacher_anchor_quality_class_weight
        ),
        teacher_region_objectness_weight=(
            recipe.teacher_region_objectness_weight
            if teacher_region_objectness_weight is None
            else teacher_region_objectness_weight
        ),
        teacher_region_class_weight=(
            recipe.teacher_region_class_weight
            if teacher_region_class_weight is None
            else teacher_region_class_weight
        ),
        teacher_region_radius_m=(
            recipe.teacher_region_radius_m
            if teacher_region_radius_m is None
            else teacher_region_radius_m
        ),
        score_threshold_candidates=(
            recipe.score_threshold_candidates
            if score_threshold_candidates is None
            else score_threshold_candidates
        ),
        top_k_candidates=(
            recipe.top_k_candidates if top_k_candidates is None else top_k_candidates
        ),
        official_eval_every_epochs=(
            recipe.official_eval_every_epochs
            if official_eval_every_epochs is None
            else official_eval_every_epochs
        ),
        official_eval_score_threshold=(
            recipe.official_eval_score_threshold
            if official_eval_score_threshold is None
            else official_eval_score_threshold
        ),
        official_eval_top_k=(
            recipe.official_eval_top_k if official_eval_top_k is None else official_eval_top_k
        ),
        early_stop_patience=(
            recipe.early_stop_patience if early_stop_patience is None else early_stop_patience
        ),
        early_stop_min_delta=(
            recipe.early_stop_min_delta
            if early_stop_min_delta is None
            else early_stop_min_delta
        ),
        early_stop_min_epochs=(
            recipe.early_stop_min_epochs
            if early_stop_min_epochs is None
            else early_stop_min_epochs
        ),
        skip_training=recipe.skip_training if skip_training is None else skip_training,
        execution_backend=(
            recipe.execution_backend if execution_backend is None else execution_backend
        ),
        external_repo_root=(
            recipe.external_repo_root if external_repo_root is None else external_repo_root
        ),
        external_config_relpath=(
            recipe.external_config_relpath
            if external_config_relpath is None
            else external_config_relpath
        ),
        external_image_tag=(
            recipe.external_image_tag if external_image_tag is None else external_image_tag
        ),
        external_version=(
            recipe.external_version if external_version is None else external_version
        ),
        comparison_tier=(
            recipe.comparison_tier if comparison_tier is None else comparison_tier
        ),
        promotion_eligible=(
            recipe.promotion_eligible if promotion_eligible is None else promotion_eligible
        ),
        frontier_claim_eligible=(
            recipe.frontier_claim_eligible
            if frontier_claim_eligible is None
            else frontier_claim_eligible
        ),
        exact_public_contract=(
            recipe.exact_public_contract if exact_public_contract is None else exact_public_contract
        ),
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
        weight_decay=float(selected.get("weight_decay", 1e-4)),
        epochs=int(selected.get("epochs", 6)),
        max_train_steps=(
            int(selected["max_train_steps"])
            if selected.get("max_train_steps") is not None
            else 960
        ),
        num_workers=int(selected.get("num_workers", 4)),
        score_threshold=float(selected.get("score_threshold", 0.05)),
        top_k=int(selected.get("top_k", 112)),
        init_checkpoint=(
            str(selected["checkpoint_path"])
            if selected.get("checkpoint_path") is not None
            else None
        ),
        optimizer_schedule=cast(
            Literal["cosine", "constant"],
            selected.get("optimizer_schedule", "cosine"),
        ),
        grad_clip_norm=(
            float(selected["grad_clip_norm"])
            if selected.get("grad_clip_norm") is not None
            else None
        ),
        keep_best_checkpoint=bool(selected.get("keep_best_checkpoint", True)),
        augmentation_mode=cast(
            Literal["off", "moderate", "strong"],
            selected.get("augmentation_mode", "off"),
        ),
        enable_teacher_distillation=bool(selected.get("enable_teacher_distillation", True)),
        loss_mode=cast(
            Literal["baseline", "focal_hardneg", "quality_focal"],
            selected.get("loss_mode", "baseline"),
        ),
        hard_negative_ratio=int(selected.get("hard_negative_ratio", 3)),
        hard_negative_cap=int(selected.get("hard_negative_cap", 96)),
        teacher_anchor_quality_class_weight=float(
            selected.get("teacher_anchor_quality_class_weight", 0.0)
        ),
        teacher_region_objectness_weight=float(
            selected.get("teacher_region_objectness_weight", 0.0)
        ),
        teacher_region_class_weight=float(selected.get("teacher_region_class_weight", 0.0)),
        teacher_region_radius_m=float(selected.get("teacher_region_radius_m", 4.0)),
        score_threshold_candidates=tuple(
            float(value) for value in selected.get("score_threshold_candidates", [0.05])
        ),
        top_k_candidates=tuple(int(value) for value in selected.get("top_k_candidates", [112])),
        official_eval_every_epochs=(
            int(selected["official_eval_every_epochs"])
            if selected.get("official_eval_every_epochs") is not None
            else None
        ),
        official_eval_score_threshold=float(
            selected.get("official_eval_score_threshold", 0.20)
        ),
        official_eval_top_k=int(selected.get("official_eval_top_k", 40)),
        early_stop_patience=(
            int(selected["early_stop_patience"])
            if selected.get("early_stop_patience") is not None
            else None
        ),
        early_stop_min_delta=float(selected.get("early_stop_min_delta", 0.0)),
        early_stop_min_epochs=int(selected.get("early_stop_min_epochs", 0)),
        execution_backend=cast(
            Literal["tsqbev_local", "bevdet_official"],
            selected.get("execution_backend", "tsqbev_local"),
        ),
        external_repo_root=cast(str | None, selected.get("external_repo_root")),
        external_config_relpath=cast(str | None, selected.get("external_config_relpath")),
        external_image_tag=cast(str | None, selected.get("external_image_tag")),
        external_version=cast(str | None, selected.get("external_version")),
    )


def _load_summary_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _summary_selected_record(path: Path) -> dict[str, Any] | None:
    payload = _load_summary_payload(path)
    if payload is None:
        return None
    selected = payload.get("selected_record")
    return selected if isinstance(selected, dict) else None


def _historical_summary_paths(artifact_root: Path) -> list[Path]:
    artifacts_root = artifact_root.parent.parent
    candidates = list(artifacts_root.glob("research_*/research_loop/summary.json"))
    current_summary = artifact_root / "summary.json"
    if current_summary.exists():
        candidates.append(current_summary)
    deduped: dict[str, Path] = {str(path.resolve()): path for path in candidates}
    return sorted(
        deduped.values(),
        key=lambda path: (
            path.stat().st_mtime if path.exists() else 0.0,
            str(path),
        ),
    )


def _historical_selected_records(
    artifact_root: Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in _historical_summary_paths(artifact_root):
        record = _summary_selected_record(summary_path)
        if record is not None:
            records.append(record)
    if limit is not None:
        return records[-limit:]
    return records


def _record_nds(record: dict[str, Any]) -> float:
    evaluation = record.get("evaluation", {})
    assert isinstance(evaluation, dict)
    return _metric_float(evaluation.get("nd_score"), float("-inf"))


def _record_map(record: dict[str, Any]) -> float:
    evaluation = record.get("evaluation", {})
    assert isinstance(evaluation, dict)
    return _metric_float(evaluation.get("mean_ap"), float("-inf"))


def _record_val_total(record: dict[str, Any]) -> float:
    val = record.get("val", {})
    assert isinstance(val, dict)
    return _metric_float(val.get("total"), float("inf"))


def _record_boxes_mean(record: dict[str, Any]) -> float:
    geometry = record.get("prediction_geometry", {})
    assert isinstance(geometry, dict)
    return _metric_float(geometry.get("boxes_per_sample_mean"), float("inf"))


def _best_historical_record(
    artifact_root: Path,
    *,
    exclude_recipe: str | None = None,
) -> dict[str, Any] | None:
    candidates = [
        record
        for record in _historical_selected_records(artifact_root)
        if exclude_recipe is None or str(record.get("recipe")) != exclude_recipe
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda record: (
            -_record_nds(record),
            -_record_map(record),
            _record_val_total(record),
        ),
    )[0]


def _boss_progress_verdict(
    *,
    current_record: dict[str, Any],
    previous_record: dict[str, Any] | None,
    artifact_root: Path,
) -> dict[str, Any]:
    current_nds = _record_nds(current_record)
    current_map = _record_map(current_record)
    current_val = _record_val_total(current_record)
    current_boxes = _record_boxes_mean(current_record)
    current_recipe = str(current_record.get("recipe", ""))

    previous_nds = None
    previous_map = None
    previous_val = None
    previous_boxes = None
    delta_vs_previous: dict[str, float] | None = None
    if previous_record is not None:
        previous_nds = _record_nds(previous_record)
        previous_map = _record_map(previous_record)
        previous_val = _record_val_total(previous_record)
        previous_boxes = _record_boxes_mean(previous_record)
        delta_vs_previous = {
            "nds": current_nds - previous_nds,
            "map": current_map - previous_map,
            "val_total": previous_val - current_val,
            "boxes_per_sample_mean": previous_boxes - current_boxes,
        }

    best_all_time = _best_historical_record(artifact_root, exclude_recipe=current_recipe)
    delta_vs_best_all_time: dict[str, float] | None = None
    best_all_time_recipe: str | None = None
    if best_all_time is not None:
        best_all_time_recipe = str(best_all_time.get("recipe"))
        delta_vs_best_all_time = {
            "nds": current_nds - _record_nds(best_all_time),
            "map": current_map - _record_map(best_all_time),
            "val_total": _record_val_total(best_all_time) - current_val,
            "boxes_per_sample_mean": _record_boxes_mean(best_all_time) - current_boxes,
        }

    progress_class = "establishing"
    reason = "no previous incumbent was available for comparison"
    if delta_vs_previous is not None:
        nds_delta = delta_vs_previous["nds"]
        map_delta = delta_vs_previous["map"]
        box_delta = delta_vs_previous["boxes_per_sample_mean"]
        if nds_delta <= -0.003 or (nds_delta < 0.0 and map_delta <= -0.01):
            progress_class = "regression"
            reason = "regressed against the previous incumbent on the main mini_val metrics"
        elif nds_delta >= 0.01 or (nds_delta >= 0.003 and map_delta >= 0.01):
            progress_class = "breakthrough"
            reason = "cleared the aggressive improvement bar against the previous incumbent"
        elif nds_delta >= 0.003 or map_delta >= 0.01 or (nds_delta >= 0.0 and box_delta >= 20.0):
            progress_class = "meaningful"
            reason = "improved either the main metrics or a key deployment-side blocker materially"
        else:
            progress_class = "stalled"
            reason = (
                "failed to clear the minimum meaningful-uplift bar against the "
                "previous incumbent"
            )

    recent_selected = _historical_selected_records(artifact_root, limit=4)
    repeated_incremental = 0
    repeated_schedule = 0
    for record in reversed(recent_selected):
        verdict = str(record.get("root_cause_verdict", ""))
        if verdict == "incremental_progress":
            repeated_incremental += 1
        else:
            break
    for record in reversed(recent_selected):
        verdict = str(record.get("root_cause_verdict", ""))
        if verdict == "schedule_checkpoint_drift":
            repeated_schedule += 1
        else:
            break

    return {
        "progress_class": progress_class,
        "reason": reason,
        "current_recipe": current_recipe,
        "current_nds": current_nds,
        "current_map": current_map,
        "current_val_total": current_val,
        "current_boxes_per_sample_mean": current_boxes,
        "previous_recipe": None if previous_record is None else previous_record.get("recipe"),
        "previous_nds": previous_nds,
        "previous_map": previous_map,
        "delta_vs_previous_incumbent": delta_vs_previous,
        "best_all_time_recipe": best_all_time_recipe,
        "delta_vs_best_all_time": delta_vs_best_all_time,
        "recent_repeated_incremental_progress": repeated_incremental,
        "recent_repeated_schedule_checkpoint_drift": repeated_schedule,
    }


def _boss_policy_from_history(
    artifact_root: Path,
    *,
    extra_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recent_selected = _historical_selected_records(artifact_root, limit=4)
    if extra_record is not None:
        recent_selected = [*recent_selected[-3:], extra_record]
    latest = recent_selected[-1] if recent_selected else None

    repeated_incremental = 0
    repeated_schedule = 0
    for record in reversed(recent_selected):
        verdict = str(record.get("root_cause_verdict", ""))
        if verdict == "incremental_progress":
            repeated_incremental += 1
        else:
            break
    for record in reversed(recent_selected):
        verdict = str(record.get("root_cause_verdict", ""))
        if verdict == "schedule_checkpoint_drift":
            repeated_schedule += 1
        else:
            break

    priority_tags: list[str] = []
    suppress_tags: list[str] = []
    force_priority_only = False
    reasons: list[str] = []
    if repeated_incremental >= 2 or repeated_schedule >= 2:
        force_priority_only = True
        suppress_tags.extend(
            ["query_boost", "lr_down", "augmentation", "focal_hardneg", "unfreeze"]
        )
        reasons.append(
            "suppressed low-ROI exploit families after repeated incremental or "
            "schedule-drift outcomes"
        )
    if latest is not None:
        config = latest.get("config", {})
        assert isinstance(config, dict)
        source_mix = latest.get("source_mix", {})
        assert isinstance(source_mix, dict)
        frontier_incumbent = str(config.get("image_backbone", "")).startswith("dinov3_")
        if frontier_incumbent:
            force_priority_only = True
            suppress_tags.extend(
                [
                    "teacher_bag",
                    "anchor_mix",
                    "query_boost",
                    "lr_down",
                    "augmentation",
                    "focal_hardneg",
                    "unfreeze",
                ]
            )
            priority_tags = [
                "quality_rank_finegrid",
                "world_aligned_distillation",
                "sam21_offline_support",
                "dino_v3",
            ]
            reasons.append(
                "the active incumbent is already on the hard-pivot DINOv3 path; keep "
                "frontier guardrail, distillation, SAM2, and backbone-capacity ablations only"
            )
        if latest.get("use_teacher_provider") and not frontier_incumbent:
            priority_tags.append("teacher_bag")
            priority_tags.append("teacher_off_control")
        if (
            not frontier_incumbent
            and
            latest.get("use_teacher_provider")
            and str(config.get("ranking_mode")) != "quality_class_only"
        ):
            priority_tags.append("quality_rank")
            reasons.append("ranking is still misaligned with the quality-aware target")
        if (
            not frontier_incumbent
            and (
                float(source_mix.get("lidar", 0.0)) >= 0.8
                or float(source_mix.get("proposal", 0.0)) < 0.1
            )
        ):
            priority_tags.append("anchor_mix")
            reasons.append("source mix is still collapsing toward LiDAR anchors")
        if (
            not frontier_incumbent
            and
            latest.get("use_teacher_provider")
            and _metric_float(latest.get("teacher_anchor_quality_class_weight"), 0.0) <= 0.0
        ):
            priority_tags.append("teacher_quality")
            reasons.append("teacher quality supervision is not yet active on the incumbent")
        if (
            not frontier_incumbent
            and
            latest.get("use_teacher_provider")
            and str(config.get("ranking_mode")) == "quality_class_only"
            and _metric_float(latest.get("teacher_anchor_quality_class_weight"), 0.0) > 0.0
        ):
            force_priority_only = True
            suppress_tags.extend(["teacher_bag", "anchor_mix", "query_boost", "lr_down"])
            priority_tags = [
                "quality_rank_finegrid",
                "teacher_quality_plus",
                "teacher_off_control",
            ]
            reasons.append(
                "the winner line is already teacher_quality + quality_rank; suppress "
                "previously losing mixed branches and focus on calibration plus one "
                "surgical teacher-quality ablation"
            )
    deduped_priority: list[str] = []
    for tag in priority_tags:
        if tag not in deduped_priority:
            deduped_priority.append(tag)
    deduped_suppress: list[str] = []
    for tag in suppress_tags:
        if tag not in deduped_suppress:
            deduped_suppress.append(tag)
    return {
        "force_priority_only": force_priority_only,
        "priority_tags": deduped_priority,
        "suppress_tags": deduped_suppress,
        "reasons": reasons,
        "recent_repeated_incremental_progress": repeated_incremental,
        "recent_repeated_schedule_checkpoint_drift": repeated_schedule,
    }


def _load_best_ratio_passing_overfit_frontier(artifact_dir: Path) -> ResearchRecipe | None:
    best_recipe: ResearchRecipe | None = None
    best_key: tuple[float, float, float, float] | None = None
    gate_roots: list[Path] = []
    local_gates_root = artifact_dir / "gates"
    if local_gates_root.exists():
        gate_roots.append(local_gates_root)
    canonical_gates_root = REPO_ROOT / "artifacts" / "gates"
    if canonical_gates_root.exists() and canonical_gates_root not in gate_roots:
        gate_roots.append(canonical_gates_root)
    if not gate_roots:
        return None
    for gates_root in gate_roots:
        for summary_path in sorted(gates_root.glob("*/overfit_gate/summary.json")):
            try:
                summary = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                continue
            gate_verdict = summary.get("gate_verdict")
            if not isinstance(gate_verdict, dict) or not bool(gate_verdict.get("passed", False)):
                continue
            checkpoint_path = (
                summary.get("selected_checkpoint_path")
                or summary.get("best_checkpoint_path")
                or summary.get("checkpoint_path")
            )
            if not isinstance(checkpoint_path, str):
                continue
            checkpoint = Path(checkpoint_path)
            if not checkpoint.exists():
                continue
            try:
                _, payload = load_model_from_checkpoint(checkpoint)
            except Exception:
                continue
            config_payload = payload.get("model_config")
            if not isinstance(config_payload, dict):
                continue
            try:
                config = ModelConfig.model_validate(config_payload)
            except Exception:
                continue
            ratio = _metric_float(gate_verdict.get("train_total_ratio"), float("inf"))
            nds = _metric_float(gate_verdict.get("nds"), -1.0)
            mean_ap = _metric_float(gate_verdict.get("mean_ap"), -1.0)
            car_ap = _metric_float(gate_verdict.get("car_ap_4m"), -1.0)
            key = (nds, mean_ap, car_ap, -ratio)
            if best_key is not None and key <= best_key:
                continue
            recipe_name = str(summary.get("recipe") or summary_path.parents[1].name)
            best_key = key
            best_recipe = ResearchRecipe(
                name=f"carryover_{recipe_name}",
                note="promote the passed overfit frontier into the next bounded mini-val loop",
                hypothesis=(
                    "the passed overfit frontier should now be measured on mini_val before any "
                    "new subset-only mutations"
                ),
                mutation_reason=(
                    "carry forward the strongest ratio-passing overfit recipe as the new mini "
                    "baseline"
                ),
                config=config,
                stage="baseline",
                parent_recipe=recipe_name,
                use_teacher_provider=config.teacher_seed_mode != "off",
                batch_size=2,
                grad_accum_steps=1,
                lr=1e-4,
                weight_decay=0.0,
                epochs=6,
                max_train_steps=960,
                num_workers=4,
                score_threshold=0.05,
                top_k=64,
                init_checkpoint=str(checkpoint),
                optimizer_schedule="constant",
                grad_clip_norm=5.0,
                keep_best_checkpoint=True,
                augmentation_mode="off",
                enable_teacher_distillation=False,
                loss_mode="quality_focal",
                hard_negative_ratio=3,
                hard_negative_cap=96,
                teacher_region_objectness_weight=0.0,
                teacher_region_radius_m=4.0,
                score_threshold_candidates=(0.05, 0.15, 0.25, 0.35),
                top_k_candidates=(16, 32, 64),
            )
    return best_recipe


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
            "geometry-aware teacher supervision should improve ranking and localization "
            "before more invasive seed mutations"
        ),
        mutation_reason="enable teacher cache supervision as a paired ablation",
        stage=stage,
        use_teacher_provider=True,
        enable_teacher_distillation=True,
    )


def _initial_recipes(
    artifact_root: Path,
    *,
    teacher_provider_available: bool = False,
    boss_policy: dict[str, Any] | None = None,
) -> list[ResearchRecipe]:
    suppress_tags = {str(tag) for tag in (boss_policy or {}).get("suppress_tags", [])}
    priority_tags = [str(tag) for tag in (boss_policy or {}).get("priority_tags", [])]
    force_priority_only = bool((boss_policy or {}).get("force_priority_only", False))

    def filter_tagged(tagged: list[tuple[str, ResearchRecipe]]) -> list[ResearchRecipe]:
        filtered = [
            (tag, recipe)
            for tag, recipe in tagged
            if tag not in suppress_tags
        ]
        if priority_tags:
            priority_order = {tag: index for index, tag in enumerate(priority_tags)}
            filtered = sorted(
                filtered,
                key=lambda item: priority_order.get(item[0], len(priority_order)),
            )
        if force_priority_only and priority_tags:
            allowed_tags = set(priority_tags) | {"baseline"}
            filtered = [
                (tag, recipe)
                for tag, recipe in filtered
                if tag in allowed_tags
            ]
        deduped: list[ResearchRecipe] = []
        seen_names: set[str] = set()
        for _tag, recipe in filtered:
            if recipe.name in seen_names:
                continue
            seen_names.add(recipe.name)
            deduped.append(recipe)
        return deduped

    if _priority_tags_request_public_student(priority_tags):
        public_primary = _public_bevdet_primary_recipe()
        public_temporal = _make_public_bevdet_temporal_recipe(public_primary)
        tagged_public: list[tuple[str, ResearchRecipe]] = [
            ("public_student_replacement", public_primary),
            ("bevdet_public_student", public_primary),
            ("camera_bev_working_baseline", public_primary),
            ("official_box_coder", public_primary),
            ("bevdepth_temporal_student", public_temporal),
            ("bevdet_public_student", public_temporal),
        ]
        recipes = filter_tagged(tagged_public)
        if recipes:
            return recipes
        if force_priority_only:
            raise RuntimeError(
                "public-student launch tags were requested, but no executable public-student "
                "recipes survived filtering"
            )

    if _priority_tags_request_frontier(priority_tags):
        use_light_bridge = any(
            tag in {"lightweight_bridge", "gated_cross_attention", "teacher_side_foundation"}
            for tag in priority_tags
        )
        frontier_vits = (
            _frontier_light_bridge_recipe(teacher_provider_available=teacher_provider_available)
            if use_light_bridge
            else _frontier_vits16_recipe(teacher_provider_available=teacher_provider_available)
        )
        frontier_tagged: list[tuple[str, ResearchRecipe]] = [
            ("dino_v3", frontier_vits),
            ("dino_v3_bridge", frontier_vits),
            ("bevformer_v2_perspective_supervision", frontier_vits),
            ("sparse4d_efficiency", frontier_vits),
            ("sam21_offline_support", frontier_vits),
            ("geometry_sanity", _make_frontier_official_guardrail_recipe(frontier_vits)),
            ("official_metric_only", _make_frontier_official_guardrail_recipe(frontier_vits)),
        ]
        if use_light_bridge:
            frontier_tagged.extend(
                [
                    ("teacher_bootstrap", frontier_vits),
                    ("lightweight_bridge", frontier_vits),
                    ("gated_cross_attention", frontier_vits),
                    ("teacher_side_foundation", frontier_vits),
                    (
                        "lightweight_bridge",
                        _make_frontier_light_bridge_effb0_recipe(frontier_vits),
                    ),
                ]
            )
        if teacher_provider_available:
            frontier_tagged.extend(
                [
                    (
                        "world_aligned_distillation",
                        _make_frontier_world_distill_recipe(frontier_vits),
                    ),
                    (
                        "world_latent_distillation",
                        _make_frontier_world_distill_recipe(frontier_vits),
                    ),
                    ("openpcdet_teacher", frontier_vits),
                    ("bevfusion_teacher", frontier_vits),
                ]
            )
        if use_light_bridge:
            frontier_tagged.append(
                ("teacher_side_foundation", _make_frontier_teacher_control_recipe(frontier_vits))
            )
        else:
            frontier_tagged.append(("dino_v3", _make_frontier_vitb16_recipe(frontier_vits)))
            frontier_tagged.append(
                ("sam21_offline_support", _make_frontier_sam2_ablation_recipe(frontier_vits))
            )
        if teacher_provider_available:
            frontier_tagged.append(
                ("world_aligned_distillation", _make_frontier_teacher_control_recipe(frontier_vits))
            )
        else:
            frontier_tagged.append(
                ("dino_v3", _make_frontier_teacher_control_recipe(frontier_vits))
            )
        recipes = filter_tagged(frontier_tagged)
        if recipes:
            return recipes
        if force_priority_only:
            raise RuntimeError(
                "frontier launch tags were requested, but no executable hard-pivot recipes "
                "survived filtering"
            )

    carryover = _load_previous_incumbent(artifact_root)
    if carryover is not None:
        query_boost = _make_query_boost_recipe(
            carryover,
            source_mix={"lidar": 0.33, "proposal": 0.50, "global": 0.17},
            stage="explore",
        )
        lr_down = _make_lr_down_recipe(carryover, stage="explore")
        tagged: list[tuple[str, ResearchRecipe]] = [("baseline", carryover)]
        if teacher_provider_available and carryover.config.teacher_seed_mode == "off":
            tagged.append(("teacher_seed", _make_teacher_seed_recipe(carryover)))
        tagged.extend(
            [
                ("query_boost", query_boost),
                ("lr_down", lr_down),
            ]
        )
        return filter_tagged(tagged)
    overfit_carryover = _load_best_ratio_passing_overfit_frontier(artifact_root.parent)
    if overfit_carryover is not None:
        return [overfit_carryover]
    baseline = _baseline_recipe()
    proposal = _proposal_heavy_recipe()
    efficientnet = _efficientnet_recipe(proposal)
    tagged = [
        ("baseline", baseline),
        ("query_boost", proposal),
        ("unfreeze", efficientnet),
    ]
    if teacher_provider_available:
        tagged.insert(1, ("teacher_seed", _make_teacher_seed_recipe(baseline)))
    return filter_tagged(tagged)


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
        loss_mode=recipe.loss_mode,
        score_threshold_candidates=(0.05, 0.15, 0.25),
        top_k_candidates=(16, 32, 64, 112),
    )


def _make_anchor_mix_recipe(
    recipe: ResearchRecipe,
    *,
    stage: Literal["baseline", "explore", "exploit"] = "exploit",
) -> ResearchRecipe:
    reserve_proposal = min(16, recipe.config.q_2d)
    reserve_global = min(8, recipe.config.q_global)
    config = _updated_config(
        recipe.config,
        anchor_first_min_proposal=reserve_proposal,
        anchor_first_min_global=reserve_global,
    )
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_anchor_mix",
        note="preserve a small non-LiDAR budget inside anchor-first routing",
        hypothesis=(
            "teacher anchors should stay primary, but reserving proposal/global slots should "
            "improve multimodal stability and suppress overconfident LiDAR-only ranking"
        ),
        mutation_reason=(
            "reserve a bounded proposal/global floor inside anchor-first routing to stop "
            "source-mix collapse"
        ),
        config=config,
        stage=stage,
        loss_mode=recipe.loss_mode,
        score_threshold_candidates=(0.05, 0.15, 0.25),
        top_k_candidates=(16, 32, 64),
    )


def _make_teacher_off_control_recipe(
    recipe: ResearchRecipe,
    *,
    stage: Literal["baseline", "explore", "exploit"] = "exploit",
) -> ResearchRecipe:
    config = _updated_config(recipe.config, teacher_seed_mode="off")
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_teacher_off_control",
        note="paired teacher-off control on the incumbent optimization regime",
        hypothesis=(
            "a true teacher-on versus teacher-off pair is required to measure whether the "
            "current gains come from teacher anchors or only from optimization drift"
        ),
        mutation_reason="disable teacher seeding while keeping the rest of the regime fixed",
        config=config,
        stage=stage,
        use_teacher_provider=False,
        enable_teacher_distillation=False,
    )


def _make_teacher_seed_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    config = _updated_config(
        recipe.config,
        teacher_seed_mode="replace_lidar",
        teacher_seed_selection_mode="class_balanced_round_robin",
        router_mode="anchor_first",
    )
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_teacher_seed",
        note="use cached external teacher-guided seeds as the anchor-first object set",
        hypothesis=(
            "a strong external LiDAR teacher should ground object anchors directly; "
            "the student should refine a class-balanced teacher anchor set instead "
            "of letting raw teacher scores overfill the bank with easy classes"
        ),
        mutation_reason=(
            "switch teacher-seeded runs to anchor-first routing, preserve a class-balanced "
            "teacher anchor set, and disable extra KD"
        ),
        config=config,
        stage="exploit",
        use_teacher_provider=True,
        enable_teacher_distillation=False,
        score_threshold_candidates=(0.05, 0.15, 0.25, 0.35),
        top_k_candidates=(16, 32, 64),
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


def _make_focal_hardneg_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_focal_hardneg",
        note="switch to focal-style ranking with bounded hard negatives",
        hypothesis=(
            "the remaining failure is ranking/overproduction, so focal hard-negative training "
            "should suppress unmatched queries while preserving teacher-grounded positives"
        ),
        mutation_reason="change the detection objective to focal hard-negative mode",
        stage="exploit",
        loss_mode="focal_hardneg",
        score_threshold_candidates=(0.05, 0.15, 0.25),
        top_k_candidates=(32, 64, 112),
    )


def _make_quality_focal_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_quality_focal",
        note="switch objectness supervision to quality-aware focal ranking",
        hypothesis=(
            "the remaining failure is ranking, not pure classification; quality-aware "
            "objectness should suppress unmatched queries while preserving teacher-grounded "
            "car hypotheses"
        ),
        mutation_reason=(
            "replace plain objectness BCE with quality focal supervision aligned to matched "
            "BEV center quality"
        ),
        stage="exploit",
        loss_mode="quality_focal",
        score_threshold_candidates=(0.05, 0.15, 0.25, 0.35),
        top_k_candidates=(16, 32, 64),
    )


def _make_quality_rank_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    config = _updated_config(recipe.config, ranking_mode="quality_class_only")
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_quality_rank",
        note="rank detections with the quality-aware class score directly",
        hypothesis=(
            "quality-aware supervision should drive the exported ranking score directly; "
            "post-hoc class*objectness products are likely suppressing good matched queries"
        ),
        mutation_reason=(
            "align export ranking with the quality-aware class target instead of multiplying "
            "separate class and objectness heads"
        ),
        config=config,
        stage="exploit",
        loss_mode="quality_focal",
        score_threshold_candidates=(0.05, 0.15, 0.25, 0.35),
        top_k_candidates=(16, 32, 64),
    )


def _make_quality_rank_finegrid_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    config = _updated_config(recipe.config, ranking_mode="quality_class_only")
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_quality_rank_finegrid",
        note="keep the winning quality-rank line and search only the export boundary",
        hypothesis=(
            "the current incumbent is already strong enough to pressure the v16 frontier; "
            "the remaining gap is a calibration boundary around the geometry gate, not a "
            "new architecture family"
        ),
        mutation_reason=(
            "run an eval-only fine grid around the proven top-k / threshold boundary instead "
            "of spending another full train cycle"
        ),
        config=config,
        stage="exploit",
        score_threshold_candidates=(0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32),
        top_k_candidates=(40, 48, 56, 64),
        skip_training=True,
    )


def _make_teacher_region_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_teacher_region",
        note="add teacher-region objectness supervision around cached teacher boxes",
        hypothesis=(
            "the student still overproduces queries because objectness is weakly tied to "
            "teacher-supported regions; a soft region prior should improve ranking without "
            "requiring denser teacher maps"
        ),
        mutation_reason=(
            "add a low-weight teacher-region objectness loss derived from cached teacher boxes "
            "and scores"
        ),
        stage="exploit",
        loss_mode="quality_focal",
        teacher_region_objectness_weight=max(recipe.teacher_region_objectness_weight, 0.15),
        teacher_region_radius_m=6.0,
        score_threshold_candidates=(0.05, 0.15, 0.25, 0.35),
        top_k_candidates=(16, 32, 48, 64),
    )


def _make_teacher_quality_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_teacher_quality",
        note="preserve teacher class-quality signal on anchored and nearby queries",
        hypothesis=(
            "the current student still loses teacher object hypotheses in ranking and class "
            "confidence; soft teacher quality targets should retain those hypotheses better "
            "than hard label-only anchor losses"
        ),
        mutation_reason=(
            "add quality-aware teacher class supervision on teacher-seeded queries plus "
            "soft class-region targets around teacher boxes"
        ),
        stage="exploit",
        loss_mode="quality_focal",
        teacher_anchor_quality_class_weight=max(
            recipe.teacher_anchor_quality_class_weight,
            0.35,
        ),
        teacher_region_objectness_weight=max(recipe.teacher_region_objectness_weight, 0.10),
        teacher_region_class_weight=max(recipe.teacher_region_class_weight, 0.10),
        teacher_region_radius_m=6.0,
        score_threshold_candidates=(0.05, 0.15, 0.25, 0.35),
        top_k_candidates=(16, 32, 48, 64),
    )


def _make_teacher_quality_plus_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    config = _updated_config(recipe.config, ranking_mode="quality_class_only")
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_teacher_quality_plus",
        note="keep the winner line fixed and raise only teacher-quality pressure slightly",
        hypothesis=(
            "the quality-rank line is the current winner; a bounded increase in teacher "
            "quality supervision may preserve more teacher confidence without the dilution "
            "introduced by anchor-mix or heavy augmentation"
        ),
        mutation_reason=(
            "raise only teacher-quality and teacher-region weights on the winning line"
        ),
        config=config,
        stage="exploit",
        teacher_anchor_quality_class_weight=max(
            recipe.teacher_anchor_quality_class_weight,
            0.45,
        ),
        teacher_region_objectness_weight=max(recipe.teacher_region_objectness_weight, 0.12),
        teacher_region_class_weight=max(recipe.teacher_region_class_weight, 0.12),
        score_threshold_candidates=(0.20, 0.24, 0.28, 0.32),
        top_k_candidates=(32, 40, 48, 56),
    )


def _make_teacher_bag_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    q_2d = min(max(recipe.config.q_2d, 80), 128)
    q_global = recipe.config.q_global
    q_lidar = recipe.config.q_lidar
    total_queries = q_lidar + q_2d + q_global
    max_object_queries = min(max(recipe.config.max_object_queries, 112), total_queries)
    config = _updated_config(
        recipe.config,
        ranking_mode="quality_class_only",
        q_2d=q_2d,
        max_object_queries=max_object_queries,
        proposals_per_view=min(max(recipe.config.proposals_per_view, 24), 32),
        anchor_first_min_proposal=max(
            recipe.config.anchor_first_min_proposal,
            min(16, q_2d),
        ),
        anchor_first_min_global=max(
            recipe.config.anchor_first_min_global,
            min(8, q_global),
        ),
    )
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_teacher_bag",
        note=(
            "bundle the proven ranking, teacher-quality, source-mix, and robustness "
            "tricks into one efficient teacher-anchored recipe"
        ),
        hypothesis=(
            "a bundled teacher-anchor recipe with quality-aware ranking, reserved "
            "non-LiDAR slots, teacher-region supervision, and label-safe augmentation "
            "should outperform piecemeal sparse mutations"
        ),
        mutation_reason=(
            "compose the current highest-ROI local tricks into one bag-of-tricks "
            "candidate instead of testing them one by one"
        ),
        config=config,
        stage="exploit",
        augmentation_mode="moderate",
        loss_mode="quality_focal",
        teacher_anchor_quality_class_weight=max(
            recipe.teacher_anchor_quality_class_weight,
            0.40,
        ),
        teacher_region_objectness_weight=max(recipe.teacher_region_objectness_weight, 0.15),
        teacher_region_class_weight=max(recipe.teacher_region_class_weight, 0.15),
        teacher_region_radius_m=6.0,
        score_threshold_candidates=(0.05, 0.10, 0.15, 0.25),
        top_k_candidates=(16, 32, 48, 64),
    )


def _make_augmented_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    next_mode: Literal["off", "moderate", "strong"]
    if recipe.augmentation_mode == "off":
        next_mode = "moderate"
    elif recipe.augmentation_mode == "moderate":
        next_mode = "strong"
    else:
        next_mode = "strong"
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_{next_mode}_aug",
        note="apply label-safe camera/LiDAR augmentation in the current regime",
        hypothesis=(
            "the current mini loop may be overfitting to seed geometry and weak camera cues; "
            "moderate photometric and LiDAR noise should improve mini-val robustness"
        ),
        mutation_reason=(
            "turn on label-safe multiview photometric distortion plus LiDAR dropout/jitter "
            "without changing box or lane geometry"
        ),
        stage="exploit",
        augmentation_mode=next_mode,
        score_threshold_candidates=(0.05, 0.15, 0.25, 0.35),
        top_k_candidates=(16, 32, 48, 64),
    )


def _make_teacher_region_aug_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        _make_teacher_region_recipe(recipe),
        name=f"{recipe.name}_teacher_region_aug",
        note="combine teacher-region objectness with label-safe camera/LiDAR augmentation",
        hypothesis=(
            "the fastest next lift is likely teacher-guided ranking plus modest robustness "
            "augmentation rather than another architecture mutation"
        ),
        mutation_reason=(
            "combine the highest-ROI ranking-side KD change with bounded label-safe "
            "augmentation in one paired exploit"
        ),
        stage="exploit",
        augmentation_mode="moderate",
    )


def _make_overfit_mode_recipe(recipe: ResearchRecipe) -> ResearchRecipe:
    return _clone_recipe(
        recipe,
        name=f"{recipe.name}_overfit_mode",
        note="overfit-oriented optimization on the incumbent architecture",
        hypothesis=(
            "the current incumbent peaks early and then degrades, so constant-LR overfit mode "
            "with a looser clip should recover stronger checkpoints"
        ),
        mutation_reason="switch to overfit-mode optimization without changing the model",
        stage="exploit",
        optimizer_schedule="constant",
        weight_decay=0.0,
        grad_clip_norm=5.0,
        keep_best_checkpoint=True,
    )


def _build_exploitation_recipes(
    incumbent_recipe: ResearchRecipe,
    incumbent_record: dict[str, Any],
    teacher_provider_config: TeacherProviderConfig | None,
    remaining_budget: int,
    *,
    boss_policy: dict[str, Any] | None = None,
) -> list[ResearchRecipe]:
    if remaining_budget <= 0:
        return []
    if incumbent_recipe.execution_backend == "bevdet_official":
        tagged_candidates: list[tuple[str, ResearchRecipe]] = [
            ("bevdet_public_student", _make_public_bevdet_primary_rerun(incumbent_recipe)),
            ("bevdepth_temporal_student", _make_public_bevdet_temporal_recipe(incumbent_recipe)),
        ]
        if boss_policy is not None:
            suppress_tags = {str(tag) for tag in boss_policy.get("suppress_tags", [])}
            priority_tags = [str(tag) for tag in boss_policy.get("priority_tags", [])]
            force_priority_only = bool(boss_policy.get("force_priority_only", False))
            if suppress_tags:
                tagged_candidates = [
                    (tag, candidate)
                    for tag, candidate in tagged_candidates
                    if tag not in suppress_tags
                ]
            if priority_tags:
                priority_order = {tag: index for index, tag in enumerate(priority_tags)}
                tagged_candidates = sorted(
                    tagged_candidates,
                    key=lambda item: priority_order.get(item[0], len(priority_order)),
                )
            if force_priority_only and priority_tags:
                allowed_tags = set(priority_tags)
                tagged_candidates = [
                    (tag, candidate)
                    for tag, candidate in tagged_candidates
                    if tag in allowed_tags
                ]
        deduped: list[ResearchRecipe] = []
        seen_names: set[str] = set()
        for _tag, candidate in tagged_candidates:
            if candidate.name in seen_names:
                continue
            seen_names.add(candidate.name)
            deduped.append(candidate)
        return deduped[:remaining_budget]
    if _is_frontier_recipe(incumbent_recipe):
        tagged_candidates: list[tuple[str, ResearchRecipe]] = [
            ("quality_rank_finegrid", _make_frontier_official_guardrail_recipe(incumbent_recipe)),
            ("geometry_sanity", _make_frontier_official_guardrail_recipe(incumbent_recipe)),
            ("official_metric_only", _make_frontier_official_guardrail_recipe(incumbent_recipe)),
        ]
        if incumbent_recipe.config.fusion_style == "gated_latent_cross_attn":
            tagged_candidates.extend(
                [
                    (
                        "lightweight_bridge",
                        _make_frontier_light_bridge_effb0_recipe(incumbent_recipe),
                    ),
                    (
                        "teacher_side_foundation",
                        _make_frontier_teacher_control_recipe(incumbent_recipe),
                    ),
                ]
            )
        else:
            tagged_candidates.extend(
                [
                    ("dino_v3", _make_frontier_vitb16_recipe(incumbent_recipe)),
                    ("dino_v3_bridge", _make_frontier_vitb16_recipe(incumbent_recipe)),
                    (
                        "sam21_offline_support",
                        _make_frontier_sam2_ablation_recipe(incumbent_recipe),
                    ),
                ]
            )
        if incumbent_recipe.use_teacher_provider:
            tagged_candidates.extend(
                [
                    (
                        "world_aligned_distillation",
                        _make_frontier_world_distill_recipe(incumbent_recipe),
                    ),
                    (
                        "world_latent_distillation",
                        _make_frontier_world_distill_recipe(incumbent_recipe),
                    ),
                    (
                        "openpcdet_teacher",
                        _make_frontier_teacher_control_recipe(incumbent_recipe),
                    ),
                ]
            )
        if boss_policy is not None:
            suppress_tags = {str(tag) for tag in boss_policy.get("suppress_tags", [])}
            priority_tags = [str(tag) for tag in boss_policy.get("priority_tags", [])]
            force_priority_only = bool(boss_policy.get("force_priority_only", False))
            if suppress_tags:
                tagged_candidates = [
                    (tag, candidate)
                    for tag, candidate in tagged_candidates
                    if tag not in suppress_tags
                ]
            if priority_tags:
                priority_order = {tag: index for index, tag in enumerate(priority_tags)}
                tagged_candidates = sorted(
                    tagged_candidates,
                    key=lambda item: priority_order.get(item[0], len(priority_order)),
                )
            if force_priority_only and priority_tags:
                allowed_tags = set(priority_tags)
                tagged_candidates = [
                    (tag, candidate)
                    for tag, candidate in tagged_candidates
                    if tag in allowed_tags
                ]
        deduped: list[ResearchRecipe] = []
        seen_names: set[str] = set()
        for _tag, candidate in tagged_candidates:
            if candidate.name in seen_names:
                continue
            seen_names.add(candidate.name)
            deduped.append(candidate)
        return deduped[:remaining_budget]
    source_mix = incumbent_record.get("source_mix", {})
    assert isinstance(source_mix, dict)
    tagged_candidates: list[tuple[str, ResearchRecipe]] = []

    def add(tag: str, candidate: ResearchRecipe) -> None:
        tagged_candidates.append((tag, candidate))

    if teacher_provider_config is not None and incumbent_recipe.config.teacher_seed_mode == "off":
        add("teacher_seed", _make_teacher_seed_recipe(incumbent_recipe))
        add("teacher_kd", _make_teacher_kd_recipe(incumbent_recipe, stage="exploit"))
    if incumbent_recipe.config.teacher_seed_mode != "off":
        add("teacher_off_control", _make_teacher_off_control_recipe(incumbent_recipe))
    if (
        incumbent_recipe.use_teacher_provider
        and incumbent_recipe.config.ranking_mode == "quality_class_only"
    ):
        add("quality_rank_finegrid", _make_quality_rank_finegrid_recipe(incumbent_recipe))
    if (
        incumbent_recipe.use_teacher_provider
        and incumbent_recipe.config.ranking_mode == "quality_class_only"
        and incumbent_recipe.teacher_anchor_quality_class_weight > 0.0
    ):
        add("teacher_quality_plus", _make_teacher_quality_plus_recipe(incumbent_recipe))
    if incumbent_recipe.use_teacher_provider:
        add("teacher_bag", _make_teacher_bag_recipe(incumbent_recipe))
        add("teacher_quality", _make_teacher_quality_recipe(incumbent_recipe))
    if (
        incumbent_recipe.use_teacher_provider
        and incumbent_recipe.teacher_region_objectness_weight <= 0.0
    ):
        add("teacher_region", _make_teacher_region_recipe(incumbent_recipe))
        add("teacher_region_aug", _make_teacher_region_aug_recipe(incumbent_recipe))
    if incumbent_recipe.loss_mode != "quality_focal":
        add("quality_focal", _make_quality_focal_recipe(incumbent_recipe))
    elif incumbent_recipe.config.ranking_mode != "quality_class_only":
        add("quality_rank", _make_quality_rank_recipe(incumbent_recipe))
    if float(source_mix.get("lidar", 0.0)) >= 0.8 or float(source_mix.get("proposal", 0.0)) < 0.2:
        add("anchor_mix", _make_anchor_mix_recipe(incumbent_recipe))
    add("query_boost", _make_query_boost_recipe(incumbent_recipe, source_mix=source_mix))
    add("lr_down", _make_lr_down_recipe(incumbent_recipe))
    if incumbent_recipe.loss_mode != "focal_hardneg":
        add("focal_hardneg", _make_focal_hardneg_recipe(incumbent_recipe))
    if incumbent_recipe.config.freeze_image_backbone:
        add("unfreeze", _make_unfreeze_recipe(incumbent_recipe))
    if incumbent_recipe.augmentation_mode != "moderate":
        add("augmentation", _make_augmented_recipe(incumbent_recipe))

    if boss_policy is not None:
        suppress_tags = {str(tag) for tag in boss_policy.get("suppress_tags", [])}
        priority_tags = [str(tag) for tag in boss_policy.get("priority_tags", [])]
        force_priority_only = bool(boss_policy.get("force_priority_only", False))
        if suppress_tags:
            tagged_candidates = [
                (tag, candidate)
                for tag, candidate in tagged_candidates
                if tag not in suppress_tags
            ]
        if priority_tags:
            priority_order = {tag: index for index, tag in enumerate(priority_tags)}
            tagged_candidates = sorted(
                tagged_candidates,
                key=lambda item: (
                    priority_order.get(item[0], len(priority_order)),
                ),
            )
        if force_priority_only and priority_tags:
            allowed_tags = set(priority_tags)
            tagged_candidates = [
                (tag, candidate)
                for tag, candidate in tagged_candidates
                if tag in allowed_tags
            ]

    deduped: list[ResearchRecipe] = []
    seen_names: set[str] = set()
    for _tag, candidate in tagged_candidates:
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
        "weight_decay": recipe.weight_decay,
        "epochs": recipe.epochs,
        "max_train_steps": recipe.max_train_steps,
        "num_workers": recipe.num_workers,
        "score_threshold": recipe.score_threshold,
        "top_k": recipe.top_k,
        "init_checkpoint": recipe.init_checkpoint,
        "optimizer_schedule": recipe.optimizer_schedule,
        "grad_clip_norm": recipe.grad_clip_norm,
        "keep_best_checkpoint": recipe.keep_best_checkpoint,
        "augmentation_mode": recipe.augmentation_mode,
        "enable_teacher_distillation": recipe.enable_teacher_distillation,
        "loss_mode": recipe.loss_mode,
        "hard_negative_ratio": recipe.hard_negative_ratio,
        "hard_negative_cap": recipe.hard_negative_cap,
        "teacher_anchor_quality_class_weight": recipe.teacher_anchor_quality_class_weight,
        "teacher_region_objectness_weight": recipe.teacher_region_objectness_weight,
        "teacher_region_class_weight": recipe.teacher_region_class_weight,
        "teacher_region_radius_m": recipe.teacher_region_radius_m,
        "score_threshold_candidates": list(recipe.score_threshold_candidates),
        "top_k_candidates": list(recipe.top_k_candidates),
        "official_eval_every_epochs": recipe.official_eval_every_epochs,
        "official_eval_score_threshold": recipe.official_eval_score_threshold,
        "official_eval_top_k": recipe.official_eval_top_k,
        "early_stop_patience": recipe.early_stop_patience,
        "early_stop_min_delta": recipe.early_stop_min_delta,
        "early_stop_min_epochs": recipe.early_stop_min_epochs,
        "skip_training": recipe.skip_training,
        "execution_backend": recipe.execution_backend,
        "external_repo_root": recipe.external_repo_root,
        "external_config_relpath": recipe.external_config_relpath,
        "external_image_tag": recipe.external_image_tag,
        "external_version": recipe.external_version,
        "comparison_tier": recipe.comparison_tier,
        "promotion_eligible": recipe.promotion_eligible,
        "frontier_claim_eligible": recipe.frontier_claim_eligible,
        "exact_public_contract": recipe.exact_public_contract,
    }


def _recipe_tracking_backbone(recipe: ResearchRecipe) -> str:
    if recipe.execution_backend == "bevdet_official":
        config_relpath = recipe.external_config_relpath or "bevdet"
        return Path(config_relpath).stem
    return recipe.config.image_backbone


def _warm_start_checkpoint_for_recipe(
    recipe: ResearchRecipe,
    incumbent_recipe: ResearchRecipe | None,
    incumbent_record: dict[str, Any] | None,
) -> str | None:
    if recipe.init_checkpoint is not None:
        return recipe.init_checkpoint
    if incumbent_recipe is None or incumbent_record is None:
        return None
    if recipe.stage != "exploit" or recipe.parent_recipe != incumbent_recipe.name:
        return None
    if recipe.execution_backend != incumbent_recipe.execution_backend:
        return None
    if recipe.execution_backend == "bevdet_official":
        if recipe.external_config_relpath != incumbent_recipe.external_config_relpath:
            return None
        checkpoint_path = incumbent_record.get("checkpoint_path")
        return str(checkpoint_path) if checkpoint_path is not None else None
    if recipe.config.image_backbone != incumbent_recipe.config.image_backbone:
        return None
    if recipe.config.model_dim != incumbent_recipe.config.model_dim:
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
                    "selected_epoch": row.get("selected_epoch", ""),
                    "best_epoch": row.get("best_epoch", ""),
                    "hypothesis": row.get("hypothesis"),
                    "mutation_reason": row.get("mutation_reason"),
                    "root_cause_verdict": row.get("root_cause_verdict", ""),
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


def _catastrophic_training_failure(train_result: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    latest_official = train_result.get("latest_official_eval")
    metrics = latest_official if isinstance(latest_official, dict) else {}
    early_stop_reason = str(train_result.get("early_stop_reason") or "")
    if "catastrophic official-eval failure" in early_stop_reason:
        return True, metrics
    nds = _metric_float(metrics.get("nuscenes_nds"), 0.0)
    mean_ap = _metric_float(metrics.get("nuscenes_map"), 0.0)
    sanity_ok = _metric_float(metrics.get("nuscenes_export_sanity_ok"), 1.0)
    if nds <= 0.0 and mean_ap <= 0.0 and sanity_ok <= 0.0:
        return True, metrics
    return False, metrics


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
    teacher_provider_config: TeacherProviderConfig | None = None,
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
    if teacher_provider_config is not None:
        command.extend(["--teacher-kind", teacher_provider_config.kind])
        if teacher_provider_config.cache_dir is not None:
            command.extend(["--teacher-cache-dir", teacher_provider_config.cache_dir])
        if teacher_provider_config.checkpoint_path is not None:
            command.extend(["--teacher-checkpoint", teacher_provider_config.checkpoint_path])
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
    proposal: ResearchProposal | None = None,
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
            teacher_provider_config=teacher_provider_config,
        ),
        "recipe": _serialize_recipe(recipe),
        "proposal": None if proposal is None else proposal.to_dict(),
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


def _root_cause_verdict(record: dict[str, Any]) -> str:
    if bool(record.get("catastrophic_failure", False)):
        return "catastrophic_geometry_failure"
    evaluation = record.get("evaluation", {})
    assert isinstance(evaluation, dict)
    train = record.get("train", {})
    assert isinstance(train, dict)
    val = record.get("val", {})
    assert isinstance(val, dict)
    geometry = record.get("prediction_geometry", {})
    assert isinstance(geometry, dict)
    label_aps = evaluation.get("label_aps", {})
    assert isinstance(label_aps, dict)

    boxes_mean = _metric_float(geometry.get("boxes_per_sample_mean"), 0.0)
    boxes_p95 = _metric_float(geometry.get("boxes_per_sample_p95"), 0.0)
    nds = _metric_float(evaluation.get("nd_score"), 0.0)
    mean_ap = _metric_float(evaluation.get("mean_ap"), 0.0)
    nonzero_classes = _count_nonzero_classes(label_aps)
    selected_epoch = _metric_int(record.get("selected_epoch"), 0)
    epochs = _metric_int(record.get("epochs"), selected_epoch)
    if boxes_mean > 40.0 or boxes_p95 > 60.0:
        return "ranking_overproduction"
    if selected_epoch > 0 and epochs > selected_epoch + 1:
        return "schedule_checkpoint_drift"
    if nds < 0.02 and mean_ap < 0.005 and nonzero_classes <= 2:
        return "vehicle_emergence_failure"
    if _metric_float(val.get("total"), 0.0) >= _metric_float(train.get("total"), 0.0) * 0.9:
        return "weak_memorization"
    return "incremental_progress"


def _select_better_record(
    current_best: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> tuple[bool, str]:
    candidate_promotion_eligible = bool(candidate.get("promotion_eligible", True))
    candidate_tier = str(candidate.get("comparison_tier", "frontier_comparable"))
    if current_best is None and not candidate_promotion_eligible:
        return True, f"establishes the first {candidate_tier} control baseline for this invocation"
    candidate_eval = candidate.get("evaluation", {})
    assert isinstance(candidate_eval, dict)
    candidate_nds = float(candidate_eval.get("nd_score", float("-inf")))
    candidate_map = float(candidate_eval.get("mean_ap", float("-inf")))
    candidate_val = float(candidate.get("val", {}).get("total", float("inf")))
    if current_best is None:
        return True, "establishes the first completed baseline for this invocation"
    current_promotion_eligible = bool(current_best.get("promotion_eligible", True))
    current_tier = str(current_best.get("comparison_tier", "frontier_comparable"))
    if not candidate_promotion_eligible:
        return False, f"{candidate_tier} run is smoke/control only and cannot replace the incumbent"
    if not current_promotion_eligible:
        return True, f"first promotion-eligible comparable run replaces the {current_tier} baseline"
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
        elif record.get("status") == "completed" and not bool(
            record.get("promotion_eligible", True)
        ):
            record["final_decision"] = "control_only"
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
        and record.get("teacher_seed_mode") in {"replace_lidar", "replace_lidar_refs"}
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
        ("teacher_seed", teacher_seed_records),
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
    if "teacher_seed" in comparisons and "teacher_ref_seed" not in comparisons:
        comparisons["teacher_ref_seed"] = comparisons["teacher_seed"]
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
    geometry = promoted_record.get("prediction_geometry", {})
    assert isinstance(geometry, dict)

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
    geometry_gate_pass = (
        _metric_float(geometry.get("boxes_per_sample_mean"), float("inf")) <= 40.0
        and _metric_float(geometry.get("boxes_per_sample_p95"), float("inf")) <= 60.0
        and _metric_float(geometry.get("ego_translation_norm_p99"), float("inf")) <= 120.0
        and _metric_float(geometry.get("ego_translation_norm_max"), float("inf")) <= 150.0
    )
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
        "geometry_sanity": {
            "passed": geometry_gate_pass,
            "reason": (
                "exported boxes stayed in a bounded local range"
                if geometry_gate_pass
                else "exported boxes are still too numerous or too far away in the ego frame"
            ),
            "boxes_per_sample_mean": _metric_float(
                geometry.get("boxes_per_sample_mean"), float("inf")
            ),
            "boxes_per_sample_p95": _metric_float(
                geometry.get("boxes_per_sample_p95"), float("inf")
            ),
            "boxes_per_sample_max": _metric_float(
                geometry.get("boxes_per_sample_max"), float("inf")
            ),
            "ego_translation_norm_p99": _metric_float(
                geometry.get("ego_translation_norm_p99"), float("inf")
            ),
            "ego_translation_norm_max": _metric_float(
                geometry.get("ego_translation_norm_max"), float("inf")
            ),
        },
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
    geometry = promoted_record.get("prediction_geometry", {})
    if isinstance(geometry, dict) and _metric_float(
        geometry.get("boxes_per_sample_mean"), float("inf")
    ) > 40.0:
        recommendations.append(
            "fix the bounded object head so exported boxes stay near their seed refs"
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
    proposal: ResearchProposal | dict[str, Any] | None = None,
    supervisor_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a bounded mini-nuScenes experiment sweep and promote one incumbent."""

    ensure_research_loop_enabled()
    root = Path(dataroot)
    artifact_root = Path(artifact_dir) / "research_loop"
    artifact_root.mkdir(parents=True, exist_ok=True)
    max_experiments = max(1, max_experiments)
    previous_selected_record = _summary_selected_record(artifact_root / "summary.json")
    pre_run_boss_policy = _boss_policy_from_history(artifact_root)
    resolved_proposal = _normalize_proposal(proposal, fallback_policy=supervisor_policy)
    if supervisor_policy is not None:
        merged_policy = dict(pre_run_boss_policy)
        merged_policy.update(supervisor_policy)
        if "suppress_tags" in pre_run_boss_policy or "suppress_tags" in supervisor_policy:
            merged_policy["suppress_tags"] = sorted(
                {
                    *[str(tag) for tag in pre_run_boss_policy.get("suppress_tags", [])],
                    *[str(tag) for tag in supervisor_policy.get("suppress_tags", [])],
                }
            )
        if "priority_tags" in pre_run_boss_policy or "priority_tags" in supervisor_policy:
            supervisor_priority_tags = [
                str(item) for item in supervisor_policy.get("priority_tags", [])
            ]
            merged_policy["priority_tags"] = [
                *supervisor_priority_tags,
                *[
                    str(tag)
                    for tag in pre_run_boss_policy.get("priority_tags", [])
                    if str(tag) not in set(supervisor_priority_tags)
                ],
            ]
        pre_run_boss_policy = merged_policy
    if resolved_proposal is not None:
        launch_policy = _proposal_policy(resolved_proposal, phase="launch")
        proposal_merged_policy = dict(pre_run_boss_policy)
        proposal_merged_policy.update(launch_policy)
        proposal_merged_policy["suppress_tags"] = sorted(
            {
                *[str(tag) for tag in pre_run_boss_policy.get("suppress_tags", [])],
                *[str(tag) for tag in launch_policy.get("suppress_tags", [])],
            }
        )
        proposal_priority_tags = [str(tag) for tag in launch_policy.get("priority_tags", [])]
        proposal_merged_policy["priority_tags"] = [
            *proposal_priority_tags,
            *[
                str(tag)
                for tag in pre_run_boss_policy.get("priority_tags", [])
                if str(tag) not in set(proposal_priority_tags)
            ],
        ]
        proposal_merged_policy["force_priority_only"] = bool(
            launch_policy.get("force_priority_only", False)
            or pre_run_boss_policy.get("force_priority_only", False)
        )
        pre_run_boss_policy = proposal_merged_policy
    pre_run_brief = safe_build_research_brief(REPO_ROOT)
    (artifact_root / "pre_run_brief.json").write_text(json.dumps(pre_run_brief, indent=2))
    if resolved_proposal is not None:
        (artifact_root / "proposal.json").write_text(
            json.dumps(resolved_proposal.to_dict(), indent=2)
        )

    records: list[dict[str, Any]] = []
    incumbent_record: dict[str, Any] | None = None
    incumbent_recipe: ResearchRecipe | None = None
    candidate_queue = _initial_recipes(
        artifact_root,
        teacher_provider_available=teacher_provider_config is not None,
        boss_policy=pre_run_boss_policy,
    )[:max_experiments]
    initial_recipe_count = len(candidate_queue)
    recipe_index = 0

    while recipe_index < len(candidate_queue) and recipe_index < max_experiments:
        recipe = candidate_queue[recipe_index]
        stop_after_current = False
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
                    _recipe_tracking_backbone(recipe),
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
                    "augmentation_mode": recipe.augmentation_mode,
                    "official_eval_every_epochs": recipe.official_eval_every_epochs,
                    "official_eval_score_threshold": recipe.official_eval_score_threshold,
                    "official_eval_top_k": recipe.official_eval_top_k,
                    "early_stop_patience": recipe.early_stop_patience,
                    "early_stop_min_delta": recipe.early_stop_min_delta,
                    "early_stop_min_epochs": recipe.early_stop_min_epochs,
                    "teacher_anchor_quality_class_weight": (
                        recipe.teacher_anchor_quality_class_weight
                    ),
                    "teacher_region_objectness_weight": recipe.teacher_region_objectness_weight,
                    "teacher_region_class_weight": recipe.teacher_region_class_weight,
                    "teacher_region_radius_m": recipe.teacher_region_radius_m,
                },
            },
        )
        print(
            "[research] "
            f"starting run_id={run_id} recipe={recipe.name} stage={recipe.stage} "
            f"backend={recipe.execution_backend} "
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
            proposal=resolved_proposal,
            extra={"run_id": run_id, "status": "started"},
        )
        try:
            durations_s: dict[str, float] = {}
            warm_start_checkpoint = _warm_start_checkpoint_for_recipe(
                recipe,
                incumbent_recipe,
                incumbent_record,
            )
            if recipe.execution_backend == "bevdet_official":
                if recipe.external_repo_root is None or recipe.external_config_relpath is None:
                    raise RuntimeError(
                        f"public student recipe {recipe.name} is missing BEVDet repo/config wiring"
                    )
                start_time = time.perf_counter()
                external_result = run_bevdet_public_student(
                    repo_root=Path(recipe.external_repo_root),
                    dataset_root=root,
                    artifact_dir=run_dir,
                    config_relpath=recipe.external_config_relpath,
                    version=recipe.external_version or "v1.0-mini",
                    epochs=recipe.epochs,
                    samples_per_gpu=recipe.batch_size,
                    workers_per_gpu=recipe.num_workers,
                    image_tag=recipe.external_image_tag or DEFAULT_BEVDET_IMAGE_TAG,
                    load_from=(
                        None if warm_start_checkpoint is None else Path(warm_start_checkpoint)
                    ),
                    split=(
                        "mini_val"
                        if (recipe.external_version or "v1.0-mini") == "v1.0-mini"
                        else "val"
                    ),
                )
                durations_s["external_public_student"] = time.perf_counter() - start_time
                evaluation = cast(dict[str, Any], external_result["evaluation"])
                prediction_geometry = cast(
                    dict[str, Any], external_result["prediction_geometry"]
                )
                record.update(
                    {
                        "status": "completed",
                        "train": {},
                        "val": {},
                        "last_train": {},
                        "last_val": {},
                        "benchmark": external_result.get("benchmark", {}),
                        "checkpoint_path": str(external_result["checkpoint_path"]),
                        "epochs": recipe.epochs,
                        "selected_epoch": recipe.epochs,
                        "best_epoch": recipe.epochs,
                        "prediction_path": str(external_result["prediction_path"]),
                        "prediction_geometry": prediction_geometry,
                        "evaluation": evaluation,
                        "calibration": {"selected": {}},
                        "source_mix": {"lidar": 0.0, "proposal": 0.0, "global": 0.0},
                        "source_mix_diagnostics": {
                            "average": {"lidar": 0.0, "proposal": 0.0, "global": 0.0},
                            "per_batch": [],
                            "batches_measured": 0,
                        },
                        "durations_s": durations_s,
                        "train_samples": 0,
                        "val_samples": 0,
                        "external_backend": "bevdet_official",
                        "external_train_log": external_result.get("train_log"),
                        "external_test_log": external_result.get("test_log"),
                    }
                )
                if (
                    _metric_float(evaluation.get("nd_score"), 0.0) <= 0.0
                    and _metric_float(evaluation.get("mean_ap"), 0.0) <= 0.0
                    and (
                        _metric_float(prediction_geometry.get("max_box_size_m"), 0.0) > 30.0
                        or _metric_float(
                            prediction_geometry.get("ego_translation_norm_p99"), 0.0
                        )
                        > 120.0
                        or _metric_float(
                            prediction_geometry.get("boxes_per_sample_mean"), 0.0
                        )
                        > 80.0
                    )
                ):
                    record["status"] = "catastrophic_stop"
                    record["interim_decision"] = "reject"
                    record["decision_reason"] = (
                        "public student official eval is still zero and geometry sanity failed"
                    )
                    record["catastrophic_failure"] = True
                    record["root_cause_verdict"] = _root_cause_verdict(record)
                    stop_after_current = True
                    raise RuntimeError("catastrophic public-student geometry failure")
                record["root_cause_verdict"] = _root_cause_verdict(record)
                better, reason = _select_better_record(incumbent_record, record)
                record["interim_decision"] = "advance" if better else "reject"
                record["decision_reason"] = reason
                record["best_so_far"] = bool(better)
                if better:
                    incumbent_record = record
                    incumbent_recipe = recipe
                tracker.summary(
                    {
                        "eval_nds": _metric_float(evaluation.get("nd_score"), 0.0),
                        "eval_map": _metric_float(evaluation.get("mean_ap"), 0.0),
                        "prediction_boxes_mean": _metric_float(
                            prediction_geometry.get("boxes_per_sample_mean", 0.0)
                        ),
                        "prediction_ego_translation_p99": _metric_float(
                            prediction_geometry.get("ego_translation_norm_p99", 0.0)
                        ),
                    }
                )
            elif recipe.skip_training:
                if warm_start_checkpoint is None:
                    raise RuntimeError(
                        f"skip_training recipe {recipe.name} requires a warm-start checkpoint"
                    )
                checkpoint_path = Path(warm_start_checkpoint)
                model, _ = load_model_from_checkpoint(checkpoint_path)
                incumbent_train = (
                    {} if incumbent_record is None else incumbent_record.get("train", {})
                )
                incumbent_val = {} if incumbent_record is None else incumbent_record.get("val", {})
                incumbent_last_train = (
                    {} if incumbent_record is None else incumbent_record.get("last_train", {})
                )
                incumbent_last_val = (
                    {} if incumbent_record is None else incumbent_record.get("last_val", {})
                )
                incumbent_selected_epoch = (
                    None
                    if incumbent_record is None
                    else incumbent_record.get("selected_epoch")
                )
                incumbent_best_epoch = (
                    None if incumbent_record is None else incumbent_record.get("best_epoch")
                )
                train_result = {
                    "selected_train": incumbent_train,
                    "selected_val": incumbent_val,
                    "last_train": incumbent_last_train,
                    "last_val": incumbent_last_val,
                    "checkpoint_path": str(checkpoint_path),
                    "selected_epoch": incumbent_selected_epoch,
                    "best_epoch": incumbent_best_epoch,
                    "train_samples": 0,
                    "val_samples": 0,
                }
                durations_s["train"] = 0.0
            else:
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
                    weight_decay=recipe.weight_decay,
                    grad_accum_steps=recipe.grad_accum_steps,
                    batch_size=recipe.batch_size,
                    num_workers=recipe.num_workers,
                    device=device,
                    teacher_provider_config=run_teacher_provider_config,
                    init_checkpoint=warm_start_checkpoint,
                    use_amp=False,
                    log_every_steps=25,
                    optimizer_schedule=recipe.optimizer_schedule,
                    grad_clip_norm=recipe.grad_clip_norm,
                    keep_best_checkpoint=recipe.keep_best_checkpoint,
                    augmentation_mode=recipe.augmentation_mode,
                    enable_teacher_distillation=recipe.enable_teacher_distillation,
                    loss_mode=recipe.loss_mode,
                    hard_negative_ratio=recipe.hard_negative_ratio,
                    hard_negative_cap=recipe.hard_negative_cap,
                    teacher_anchor_quality_class_weight=recipe.teacher_anchor_quality_class_weight,
                    teacher_region_objectness_weight=recipe.teacher_region_objectness_weight,
                    teacher_region_class_weight=recipe.teacher_region_class_weight,
                    teacher_region_radius_m=recipe.teacher_region_radius_m,
                    early_stop_patience=recipe.early_stop_patience,
                    early_stop_min_delta=recipe.early_stop_min_delta,
                    early_stop_min_epochs=recipe.early_stop_min_epochs,
                    official_eval_every_epochs=recipe.official_eval_every_epochs,
                    official_eval_score_threshold=recipe.official_eval_score_threshold,
                    official_eval_top_k=recipe.official_eval_top_k,
                    tracker=tracker,
                )
                durations_s["train"] = time.perf_counter() - start_time
                checkpoint_path = Path(str(train_result["checkpoint_path"]))

            catastrophic_failure, catastrophic_metrics = _catastrophic_training_failure(
                cast(dict[str, Any], train_result)
            )
            if catastrophic_failure:
                record.update(
                    {
                        "status": "catastrophic_stop",
                        "interim_decision": "reject",
                        "decision_reason": (
                            "catastrophic official-eval failure; stop this invocation and "
                            "reassess before spending more GPU"
                        ),
                        "train": train_result.get("selected_train", {}),
                        "val": train_result.get("selected_val", {}),
                        "last_train": train_result.get("last_train", {}),
                        "last_val": train_result.get("last_val", {}),
                        "checkpoint_path": str(checkpoint_path),
                        "epochs": train_result.get("epochs"),
                        "selected_epoch": train_result.get("selected_epoch"),
                        "best_epoch": train_result.get("best_epoch"),
                        "latest_official_eval": catastrophic_metrics,
                        "early_stop_triggered": bool(
                            train_result.get("early_stop_triggered", False)
                        ),
                        "early_stop_reason": train_result.get("early_stop_reason"),
                        "catastrophic_failure": True,
                        "evaluation": {
                            "nd_score": _metric_float(
                                catastrophic_metrics.get("nuscenes_nds"),
                                0.0,
                            ),
                            "mean_ap": _metric_float(
                                catastrophic_metrics.get("nuscenes_map"),
                                0.0,
                            ),
                            "label_aps": {},
                            "tp_errors": {},
                        },
                        "prediction_geometry": {
                            "boxes_per_sample_mean": _metric_float(
                                catastrophic_metrics.get("nuscenes_boxes_per_sample"), 0.0
                            ),
                            "boxes_per_sample_p95": _metric_float(
                                catastrophic_metrics.get("nuscenes_boxes_per_sample"), 0.0
                            ),
                            "boxes_per_sample_max": _metric_float(
                                catastrophic_metrics.get("nuscenes_boxes_per_sample"), 0.0
                            ),
                            "ego_translation_norm_p99": _metric_float(
                                catastrophic_metrics.get("nuscenes_ego_translation_norm_p99"),
                                0.0,
                            ),
                            "ego_translation_norm_max": _metric_float(
                                catastrophic_metrics.get("nuscenes_ego_translation_norm_max"),
                                0.0,
                            ),
                            "max_box_size_m": _metric_float(
                                catastrophic_metrics.get("nuscenes_max_box_size_m"),
                                0.0,
                            ),
                            "score_mean": _metric_float(
                                catastrophic_metrics.get("nuscenes_score_mean"),
                                0.0,
                            ),
                        },
                        "source_mix": {"lidar": 0.0, "proposal": 0.0, "global": 0.0},
                        "durations_s": durations_s,
                        "train_samples": train_result.get("train_samples", 0),
                        "val_samples": train_result.get("val_samples", 0),
                    }
                )
                record["root_cause_verdict"] = _root_cause_verdict(record)
                stop_after_current = True
                raise RuntimeError(
                    str(train_result.get("early_stop_reason") or "catastrophic_stop")
                )
            model, _ = load_model_from_checkpoint(checkpoint_path)

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

            start_time = time.perf_counter()
            calibration = export_and_evaluate_nuscenes_grid(
                model=model,
                dataroot=root,
                version="v1.0-mini",
                split="mini_val",
                output_dir=run_dir / "mini_calibration",
                score_threshold_candidates=recipe.score_threshold_candidates,
                top_k_candidates=recipe.top_k_candidates,
                device=device,
                teacher_provider_config=run_teacher_provider_config,
            )
            durations_s["export_evaluate"] = time.perf_counter() - start_time

            selected_calibration = calibration["selected"]
            assert isinstance(selected_calibration, dict)
            prediction_path = Path(str(selected_calibration["prediction_path"]))
            evaluation = selected_calibration["evaluation"]
            assert isinstance(evaluation, dict)
            prediction_geometry = prediction_geometry_diagnostics(
                prediction_path,
                dataroot=root,
                version="v1.0-mini",
            )

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
                    "train": train_result["selected_train"],
                    "val": train_result["selected_val"],
                    "last_train": train_result["last_train"],
                    "last_val": train_result["last_val"],
                    "benchmark": bench,
                    "checkpoint_path": str(checkpoint_path),
                    "epochs": train_result.get("epochs"),
                    "selected_epoch": train_result.get("selected_epoch"),
                    "best_epoch": train_result.get("best_epoch"),
                    "prediction_path": str(prediction_path),
                    "prediction_geometry": prediction_geometry,
                    "evaluation": evaluation,
                    "calibration": calibration,
                    "source_mix": source_mix_diagnostics["average"],
                    "source_mix_diagnostics": source_mix_diagnostics,
                    "durations_s": durations_s,
                    "train_samples": train_samples,
                    "val_samples": val_samples,
                }
            )
            record["root_cause_verdict"] = _root_cause_verdict(record)
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
                    "prediction_boxes_mean": _metric_float(
                        prediction_geometry.get("boxes_per_sample_mean", 0.0)
                    ),
                    "prediction_ego_translation_p99": _metric_float(
                        prediction_geometry.get("ego_translation_norm_p99", 0.0)
                    ),
                    "selected_epoch": _metric_int(train_result.get("selected_epoch"), 0),
                    "best_epoch": _metric_int(train_result.get("best_epoch"), 0),
                    "calibrated_score_threshold": _metric_float(
                        selected_calibration.get("score_threshold"), 0.0
                    ),
                    "calibrated_top_k": _metric_int(selected_calibration.get("top_k"), 0),
                },
                step=epochs_run,
            )
        except Exception as exc:
            if record.get("status") != "catastrophic_stop":
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
            proposal=resolved_proposal,
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

        if stop_after_current:
            break

        if (
            recipe_index + 1 == initial_recipe_count
            and incumbent_record is not None
            and incumbent_recipe is not None
        ):
            remaining_budget = max_experiments - len(candidate_queue)
            existing_recipe_names = {candidate.name for candidate in candidate_queue}
            exploitation_candidates = _build_exploitation_recipes(
                incumbent_recipe,
                incumbent_record,
                teacher_provider_config,
                remaining_budget,
                boss_policy=(
                    {
                        **pre_run_boss_policy,
                        **_proposal_policy(resolved_proposal, phase="exploit"),
                    }
                    if resolved_proposal is not None
                    else pre_run_boss_policy
                ),
            )
            candidate_queue.extend(
                [
                    candidate
                    for candidate in exploitation_candidates
                    if candidate.name not in existing_recipe_names
                ][:remaining_budget]
            )
        recipe_index += 1

    promoted_run_id = (
        int(incumbent_record["run_id"])
        if incumbent_record is not None and bool(incumbent_record.get("promotion_eligible", True))
        else None
    )
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
    if not bool(incumbent_record.get("promotion_eligible", True)):
        smoke_summary: dict[str, Any] = {
            "status": "smoke_only",
            "reference_workflow": "karpathy/autoresearch",
            "selected_recipe": incumbent_recipe.name,
            "selected_record": incumbent_record,
            "records_path": str(artifact_root / "results.jsonl"),
            "results_tsv_path": str(artifact_root / "results.tsv"),
            "evaluation": incumbent_record.get("evaluation", {}),
            "leaderboard": _leaderboard(records),
            "relaunch": {
                "should_relaunch": True,
                "reason": (
                    "only smoke/control baselines were produced; exact public reproduction or "
                    "a promotion-eligible comparable run is still required"
                ),
            },
            "recommended_next_steps": [
                "treat this invocation as smoke/control only",
                "do not reset the frontier from a non-comparable mini run",
                "fix exact checkpoint/config/data-contract reproduction before claiming progress",
            ],
            "proposal": None if resolved_proposal is None else resolved_proposal.to_dict(),
            "supervisor_policy": supervisor_policy,
            "pre_run_brief_path": str(artifact_root / "pre_run_brief.json"),
        }
        (artifact_root / "summary.json").write_text(
            json.dumps(smoke_summary, indent=2, default=str)
        )
        memory_sync = safe_sync_research_memory(REPO_ROOT)
        post_run_brief = safe_build_research_brief(REPO_ROOT, persist_log=True)
        smoke_summary["memory_sync"] = memory_sync
        smoke_summary["post_run_brief"] = post_run_brief
        (artifact_root / "summary.json").write_text(
            json.dumps(smoke_summary, indent=2, default=str)
        )
        return smoke_summary

    scale_verdict = _scale_gate_verdict(incumbent_record, records)
    boss_progress_verdict = _boss_progress_verdict(
        current_record=incumbent_record,
        previous_record=previous_selected_record,
        artifact_root=artifact_root,
    )
    boss_policy_next = _boss_policy_from_history(artifact_root, extra_record=incumbent_record)
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
        "boss_progress_verdict": boss_progress_verdict,
        "boss_policy_pre_run": pre_run_boss_policy,
        "proposal": None if resolved_proposal is None else resolved_proposal.to_dict(),
        "supervisor_policy": supervisor_policy,
        "boss_policy_next": boss_policy_next,
        "relaunch": {
            "should_relaunch": not bool(scale_verdict.get("authorized", False)),
            "reason": (
                "scale gate not yet authorized; continue bounded proposal-driven iteration"
                if not bool(scale_verdict.get("authorized", False))
                else "scale gate authorized; stop bounded local loop and reassess"
            ),
            "next_proposal": None if resolved_proposal is None else resolved_proposal.to_dict(),
        },
        "recommended_next_steps": _recommended_next_steps(
            scale_verdict,
            incumbent_record,
            teacher_provider_config,
        ),
        "recipes": [_serialize_recipe(recipe) for recipe in candidate_queue[:max_experiments]],
        "ranked_recipe_names": [str(record.get("recipe")) for record in ranked],
        "pre_run_brief_path": str(artifact_root / "pre_run_brief.json"),
    }
    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    memory_sync = safe_sync_research_memory(REPO_ROOT)
    post_run_brief = safe_build_research_brief(REPO_ROOT, persist_log=True)
    summary["memory_sync"] = memory_sync
    summary["post_run_brief"] = post_run_brief
    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
