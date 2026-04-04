"""CLI for smoke checks, deployment validation, and real public baselines.

References:
- HotBEV deployment smoke-test discipline:
  https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf
- Karpathy autoresearch workflow staging:
  https://github.com/karpathy/autoresearch
- nuScenes official local evaluation:
  https://github.com/nutonomy/nuscenes-devkit
- OpenLane official evaluation kit:
  https://github.com/OpenDriveLab/OpenLane
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from rich import print

from tsqbev.bevfusion_env import (
    bevfusion_official_commands,
    check_bevfusion_environment,
    render_bevfusion_runbook_markdown,
)
from tsqbev.checkpoints import load_model_from_checkpoint
from tsqbev.config import ModelConfig
from tsqbev.data_checks import check_nuscenes_root, check_openlane_root
from tsqbev.eval_nuscenes import evaluate_nuscenes_predictions, export_nuscenes_predictions
from tsqbev.eval_openlane import evaluate_openlane_predictions, export_openlane_predictions
from tsqbev.export import export_core_to_onnx
from tsqbev.gap_analysis import analyze_reset_gap
from tsqbev.latency import LatencyPredictor, features_from_config
from tsqbev.maintenance_supervisor import run_maintenance_once, run_maintenance_supervisor
from tsqbev.model import TSQBEVModel
from tsqbev.openlane_download import (
    download_openlanev2_archive,
    openlanev2_archives,
    resolve_archive_keys,
)
from tsqbev.openlane_prepare import prepare_openlane_v1_from_raw
from tsqbev.openpcdet_env import check_openpcdet_environment
from tsqbev.overfit import run_nuscenes_overfit_gate
from tsqbev.research import run_bounded_research_loop
from tsqbev.research_memory import (
    build_research_brief,
    check_research_memory_health,
    manage_research_memory_services,
    query_research_memory,
    sync_research_memory,
)
from tsqbev.research_supervisor import run_research_supervisor
from tsqbev.reset_stack import recommended_reset_plan, render_reset_plan_markdown, upstream_registry
from tsqbev.runtime import benchmark_forward, run_eval_step, run_train_step
from tsqbev.synthetic import make_synthetic_batch
from tsqbev.teacher_audit import audit_nuscenes_teacher_cache
from tsqbev.teacher_backends import TeacherProviderConfig
from tsqbev.teacher_import import cache_nuscenes_detection_results
from tsqbev.train import fit_joint_public, fit_nuscenes, fit_openlane
from tsqbev.trt import run_trt_benchmark
from tsqbev.upstream_baselines import local_upstream_baselines, upstream_baselines
from tsqbev.upstream_readiness import check_upstream_stack


def smoke() -> None:
    config = ModelConfig.small()
    batch = make_synthetic_batch(config, batch_size=1)
    model = TSQBEVModel(config)
    outputs = model(batch)
    print(
        {
            "object_logits": tuple(outputs["object_logits"].shape),
            "object_boxes": tuple(outputs["object_boxes"].shape),
            "lane_logits": tuple(outputs["lane_logits"].shape),
            "lane_polylines": tuple(outputs["lane_polylines"].shape),
        }
    )


def export_smoke() -> None:
    config = ModelConfig.small()
    model = TSQBEVModel(config)
    output = export_core_to_onnx(model.core, config, Path("artifacts/export/core.onnx"))
    print({"onnx_path": str(output)})


def latency_smoke() -> None:
    config = ModelConfig()
    predictor = LatencyPredictor()
    features = features_from_config(
        config, params_m=55.0, active_pillars=1200.0, activations_mb=220.0
    )
    prediction = predictor.predict_ms(features)
    print({"predicted_ms": prediction, "production_pass": predictor.production_pass(prediction)})


def train_step_smoke() -> None:
    print(run_train_step(ModelConfig.small(), batch_size=1))


def eval_smoke() -> None:
    print(run_eval_step(ModelConfig.small(), batch_size=1))


def bench_smoke() -> None:
    print(benchmark_forward(ModelConfig.small(), steps=3, warmup=1, batch_size=1))


def trt_bench_smoke() -> None:
    print(
        run_trt_benchmark(
            ModelConfig(),
            image_height=256,
            image_width=704,
            warmup=10,
            steps=50,
        )
    )


def reset_stack_report(report_format: str) -> None:
    if report_format == "markdown":
        print(render_reset_plan_markdown())
        return
    plan = recommended_reset_plan()
    print(plan.to_dict())


def reset_gap_report() -> None:
    print(analyze_reset_gap().to_dict())


def upstream_registry_report() -> None:
    print([component.to_dict() for component in upstream_registry()])


def upstream_stack_report(projects_root: Path) -> None:
    print([status.to_dict() for status in check_upstream_stack(projects_root)])


def upstream_baselines_report(projects_root: Path) -> None:
    print(
        {
            "manifest": [baseline.to_dict() for baseline in upstream_baselines()],
            "local_status": [
                status.to_dict() for status in local_upstream_baselines(projects_root)
            ],
        }
    )


def bevfusion_env_report(
    repo_root: Path,
    dataset_root: Path,
    image_tag: str,
    gpu_count: int,
) -> None:
    print(
        {
            "status": check_bevfusion_environment(
                repo_root=repo_root,
                dataset_root=dataset_root,
            ).to_dict(),
            "commands": bevfusion_official_commands(
                repo_root=repo_root,
                dataset_root=dataset_root,
                image_tag=image_tag,
                gpu_count=gpu_count,
            ),
        }
    )


def bevfusion_runbook_report(
    repo_root: Path,
    dataset_root: Path,
    image_tag: str,
    gpu_count: int,
    report_format: str,
) -> None:
    if report_format == "markdown":
        print(
            render_bevfusion_runbook_markdown(
                repo_root=repo_root,
                dataset_root=dataset_root,
                image_tag=image_tag,
                gpu_count=gpu_count,
            )
        )
        return
    bevfusion_env_report(
        repo_root=repo_root,
        dataset_root=dataset_root,
        image_tag=image_tag,
        gpu_count=gpu_count,
    )


def list_openlanev2_archives_report() -> None:
    print(openlanev2_archives())


def download_openlanev2_report(
    archive_keys: list[str] | None,
    output_dir: Path,
    extract: bool,
    overwrite: bool,
) -> None:
    results = [
        download_openlanev2_archive(
            archive_key,
            output_dir=output_dir,
            extract=extract,
            overwrite=overwrite,
        )
        for archive_key in resolve_archive_keys(archive_keys)
    ]
    print(results)


def prepare_openlanev1_report(
    dataset_root: Path,
    include_lane3d_1000: bool,
    include_scene: bool,
    include_cipo: bool,
    force_reextract: bool,
) -> None:
    print(
        prepare_openlane_v1_from_raw(
            dataset_root,
            include_lane3d_1000=include_lane3d_1000,
            include_scene=include_scene,
            include_cipo=include_cipo,
            force_reextract=force_reextract,
        )
    )


def memory_health_report() -> None:
    print(check_research_memory_health())


def memory_backfill_report() -> None:
    print(sync_research_memory())


def memory_query_report(query: str, limit: int) -> None:
    print(query_research_memory(query, limit=limit))


def research_brief_report() -> None:
    print(build_research_brief().to_dict())


def research_sync_report() -> None:
    print(sync_research_memory())


def research_report() -> None:
    print(build_research_brief(persist_log=True).to_dict())


def memory_service_report(action: Literal["up", "down"]) -> None:
    print(manage_research_memory_services(action))


def _model_for_export(default_config: ModelConfig, checkpoint: Path | None) -> TSQBEVModel:
    if checkpoint is None:
        return TSQBEVModel(default_config)
    model, _ = load_model_from_checkpoint(checkpoint, default_config=default_config)
    return model


def _resolve_config(args: argparse.Namespace) -> ModelConfig:
    if args.preset == "small":
        config = ModelConfig.small()
    elif args.preset == "rtx5000-nuscenes-teacher":
        config = ModelConfig.rtx5000_nuscenes_teacher_bootstrap()
    elif args.preset == "rtx5000-nuscenes-dinov2-teacher":
        config = ModelConfig.rtx5000_nuscenes_dinov2_teacher()
    elif args.preset == "rtx5000-nuscenes-teacher-quality-plus":
        config = ModelConfig.rtx5000_nuscenes_teacher_quality_plus()
    elif args.preset == "rtx5000-nuscenes-query-boost":
        config = ModelConfig.rtx5000_nuscenes_query_boost()
    elif args.preset == "rtx5000-nuscenes":
        config = ModelConfig.rtx5000_nuscenes_baseline()
    else:
        config = ModelConfig()

    updates: dict[str, object] = {}
    if args.image_backbone is not None:
        updates["image_backbone"] = args.image_backbone
    if args.pretrained_image_backbone is not None:
        updates["pretrained_image_backbone"] = args.pretrained_image_backbone
    if args.freeze_image_backbone is not None:
        updates["freeze_image_backbone"] = args.freeze_image_backbone
    if args.foundation_repo_root is not None:
        updates["foundation_repo_root"] = str(args.foundation_repo_root)
    if args.activation_checkpointing is not None:
        updates["activation_checkpointing"] = args.activation_checkpointing
    if args.attention_backend is not None:
        updates["attention_backend"] = args.attention_backend
    if args.teacher_seed_mode is not None:
        updates["teacher_seed_mode"] = args.teacher_seed_mode
    if args.teacher_seed_selection_mode is not None:
        updates["teacher_seed_selection_mode"] = args.teacher_seed_selection_mode
    return config.model_copy(update=updates)


def _resolve_nuscenes_eval_split(version: str, split: str | None) -> str:
    if split is not None:
        return split
    return "mini_val" if version == "v1.0-mini" else "val"


def _resolve_teacher_provider_config(args: argparse.Namespace) -> TeacherProviderConfig | None:
    teacher_kind = args.teacher_kind
    if teacher_kind is None and args.teacher_cache_dir is not None:
        teacher_kind = "cache"
    if teacher_kind is None:
        return None
    return TeacherProviderConfig(
        kind=teacher_kind,
        cache_dir=str(args.teacher_cache_dir) if args.teacher_cache_dir is not None else None,
        checkpoint_path=(
            str(args.teacher_checkpoint) if args.teacher_checkpoint is not None else None
        ),
    )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tsqbev-poc utilities")
    parser.add_argument(
        "command",
        choices=(
            "smoke",
            "train-step",
            "eval",
            "bench",
            "trt-bench",
            "export-smoke",
            "latency",
            "check-data",
            "train-nuscenes",
            "train-openlane",
            "train-joint-public",
            "overfit-nuscenes",
            "cache-teacher-nuscenes",
            "audit-teacher-cache-nuscenes",
            "check-openpcdet-env",
            "export-nuscenes",
            "eval-nuscenes",
            "export-openlane",
            "eval-openlane",
            "research-loop",
            "research-supervisor",
            "maintenance-once",
            "maintenance-supervisor",
            "reset-stack",
            "reset-gap-report",
            "upstream-registry",
            "check-upstream-stack",
            "upstream-baselines",
            "check-bevfusion-env",
            "bevfusion-runbook",
            "list-openlanev2-archives",
            "download-openlanev2",
            "prepare-openlanev1",
            "memory-up",
            "memory-health",
            "memory-down",
            "memory-backfill",
            "memory-query",
            "research-brief",
            "research-sync",
            "research-report",
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--lane-dataset-root", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/eval/predictions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=(
            "default",
            "small",
            "rtx5000-nuscenes",
            "rtx5000-nuscenes-query-boost",
            "rtx5000-nuscenes-teacher",
            "rtx5000-nuscenes-teacher-quality-plus",
            "rtx5000-nuscenes-dinov2-teacher",
        ),
        default="default",
    )
    parser.add_argument(
        "--image-backbone",
        choices=(
            "tiny",
            "mobilenet_v3_large",
            "efficientnet_b0",
            "dinov2_vits14_reg",
            "dinov2_vitb14_reg",
        ),
        default=None,
    )
    parser.add_argument(
        "--pretrained-image-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--freeze-image-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--foundation-repo-root", type=Path, default=None)
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "math", "flash", "efficient", "cudnn"),
        default=None,
    )
    parser.add_argument(
        "--teacher-seed-mode",
        choices=("off", "replace_lidar", "replace_lidar_refs"),
        default=None,
    )
    parser.add_argument(
        "--teacher-seed-selection-mode",
        choices=("score_topk", "class_balanced_round_robin"),
        default=None,
    )
    parser.add_argument("--openlane-repo-root", type=Path, default=Path("/tmp/OpenLane"))
    parser.add_argument(
        "--bevfusion-repo-root",
        type=Path,
        default=Path("/home/achbogga/projects/bevfusion"),
    )
    parser.add_argument("--docker-image-tag", type=str, default="tsqbev-bevfusion-official:latest")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--subset", type=str, default="lane3d_300")
    parser.add_argument("--test-list", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--augmentation-mode",
        choices=("off", "moderate", "strong"),
        default="off",
    )
    parser.add_argument(
        "--optimizer-schedule",
        choices=("cosine", "constant"),
        default=None,
    )
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument(
        "--keep-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--teacher-distillation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--loss-mode",
        choices=("baseline", "focal_hardneg", "quality_focal"),
        default="baseline",
    )
    parser.add_argument("--hard-negative-ratio", type=int, default=3)
    parser.add_argument("--hard-negative-cap", type=int, default=96)
    parser.add_argument("--teacher-anchor-class-weight", type=float, default=0.5)
    parser.add_argument("--teacher-anchor-quality-class-weight", type=float, default=0.0)
    parser.add_argument("--teacher-anchor-objectness-weight", type=float, default=0.5)
    parser.add_argument("--teacher-region-objectness-weight", type=float, default=0.0)
    parser.add_argument("--teacher-region-class-weight", type=float, default=0.0)
    parser.add_argument("--teacher-region-radius-m", type=float, default=4.0)
    parser.add_argument("--lane-batch-multiplier", type=float, default=1.0)
    parser.add_argument("--official-eval-every-epochs", type=int, default=None)
    parser.add_argument("--official-eval-score-threshold", type=float, default=0.20)
    parser.add_argument("--official-eval-top-k", type=int, default=40)
    parser.add_argument("--teacher-anchor-final-class-weight", type=float, default=None)
    parser.add_argument("--teacher-anchor-final-objectness-weight", type=float, default=None)
    parser.add_argument("--teacher-anchor-bootstrap-epochs", type=int, default=0)
    parser.add_argument("--teacher-anchor-decay-epochs", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--early-stop-min-epochs", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-experiments", type=int, default=5)
    parser.add_argument("--max-invocations", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=int, default=30)
    parser.add_argument("--wait-poll-seconds", type=int, default=20)
    parser.add_argument("--interval-hours", type=int, default=24)
    parser.add_argument(
        "--git-publish",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--git-remote", type=str, default="origin")
    parser.add_argument("--git-branch", type=str, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--score-threshold-candidates", type=float, nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--top-k-candidates", type=int, nargs="*", default=None)
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--max-audit-samples", type=int, default=None)
    parser.add_argument("--scene-name", type=str, default=None)
    parser.add_argument(
        "--teacher-kind",
        choices=("cache", "openpcdet-centerpoint-pointpillar", "openpcdet-centerpoint-voxel"),
        default=None,
    )
    parser.add_argument("--teacher-cache-dir", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=Path("/home/achbogga/projects"),
    )
    parser.add_argument(
        "--report-format",
        choices=("json", "markdown"),
        default="json",
    )
    parser.add_argument(
        "--openpcdet-repo-root",
        type=Path,
        default=Path("/home/achbogga/projects/OpenPCDet_official"),
    )
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--archive-key", type=str, nargs="*", default=None)
    parser.add_argument(
        "--extract-openlanev2",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite-download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include-lane3d-1000",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include-scene",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include-cipo",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--force-reextract",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    if args.command == "smoke":
        smoke()
        return
    if args.command == "train-step":
        train_step_smoke()
        return
    if args.command == "eval":
        eval_smoke()
        return
    if args.command == "bench":
        bench_smoke()
        return
    if args.command == "trt-bench":
        trt_bench_smoke()
        return
    if args.command == "export-smoke":
        export_smoke()
        return
    if args.command == "latency":
        latency_smoke()
        return
    if args.command == "reset-stack":
        reset_stack_report(args.report_format)
        return
    if args.command == "reset-gap-report":
        reset_gap_report()
        return
    if args.command == "upstream-registry":
        upstream_registry_report()
        return
    if args.command == "check-upstream-stack":
        upstream_stack_report(args.projects_root)
        return
    if args.command == "upstream-baselines":
        upstream_baselines_report(args.projects_root)
        return
    if args.command == "check-bevfusion-env":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for check-bevfusion-env")
        bevfusion_env_report(
            repo_root=args.bevfusion_repo_root,
            dataset_root=args.dataset_root,
            image_tag=args.docker_image_tag,
            gpu_count=args.num_gpus,
        )
        return
    if args.command == "bevfusion-runbook":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for bevfusion-runbook")
        bevfusion_runbook_report(
            repo_root=args.bevfusion_repo_root,
            dataset_root=args.dataset_root,
            image_tag=args.docker_image_tag,
            gpu_count=args.num_gpus,
            report_format=args.report_format,
        )
        return
    if args.command == "list-openlanev2-archives":
        list_openlanev2_archives_report()
        return
    if args.command == "download-openlanev2":
        download_openlanev2_report(
            archive_keys=args.archive_key,
            output_dir=args.output_dir,
            extract=args.extract_openlanev2,
            overwrite=args.overwrite_download,
        )
        return
    if args.command == "prepare-openlanev1":
        dataset_root = args.dataset_root
        if dataset_root is None:
            dataset_root = Path("/mnt/storage/research/openlanev1_openxlab/OpenDriveLab___OpenLane")
        prepare_openlanev1_report(
            dataset_root=dataset_root,
            include_lane3d_1000=args.include_lane3d_1000,
            include_scene=args.include_scene,
            include_cipo=args.include_cipo,
            force_reextract=args.force_reextract,
        )
        return
    if args.command == "memory-up":
        memory_service_report("up")
        return
    if args.command == "memory-health":
        memory_health_report()
        return
    if args.command == "memory-down":
        memory_service_report("down")
        return
    if args.command == "memory-backfill":
        memory_backfill_report()
        return
    if args.command == "memory-query":
        if args.query is None:
            raise ValueError("--query is required for memory-query")
        memory_query_report(args.query, args.limit)
        return
    if args.command == "research-brief":
        research_brief_report()
        return
    if args.command == "research-sync":
        research_sync_report()
        return
    if args.command == "research-report":
        research_report()
        return
    if args.command == "check-data":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for check-data")
        print(
            {
                "nuscenes": check_nuscenes_root(args.dataset_root),
                "openlane": check_openlane_root(args.dataset_root),
            }
        )
        return
    if args.command == "train-nuscenes":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for train-nuscenes")
        config = _resolve_config(args)
        val_split = _resolve_nuscenes_eval_split(args.version, args.split)
        teacher_provider_config = _resolve_teacher_provider_config(args)
        print(
            fit_nuscenes(
                dataroot=args.dataset_root,
                artifact_dir=args.artifact_dir / "nuscenes",
                config=config,
                version=args.version,
                train_split=args.train_split,
                val_split=val_split,
                epochs=args.epochs,
                max_train_steps=args.max_train_steps,
                init_checkpoint=args.init_checkpoint,
                lr=args.lr,
                weight_decay=args.weight_decay,
                optimizer_schedule=(
                    args.optimizer_schedule if args.optimizer_schedule is not None else "cosine"
                ),
                grad_clip_norm=1.0 if args.grad_clip_norm is None else args.grad_clip_norm,
                keep_best_checkpoint=args.keep_best_checkpoint,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=(
                    args.early_stop_min_delta if args.early_stop_min_delta is not None else 0.0
                ),
                early_stop_min_epochs=(
                    args.early_stop_min_epochs if args.early_stop_min_epochs is not None else 0
                ),
                augmentation_mode=args.augmentation_mode,
                loss_mode=args.loss_mode,
                hard_negative_ratio=args.hard_negative_ratio,
                hard_negative_cap=args.hard_negative_cap,
                teacher_anchor_class_weight=args.teacher_anchor_class_weight,
                teacher_anchor_quality_class_weight=args.teacher_anchor_quality_class_weight,
                teacher_anchor_objectness_weight=args.teacher_anchor_objectness_weight,
                teacher_region_objectness_weight=args.teacher_region_objectness_weight,
                teacher_region_class_weight=args.teacher_region_class_weight,
                teacher_region_radius_m=args.teacher_region_radius_m,
                teacher_anchor_final_class_weight=args.teacher_anchor_final_class_weight,
                teacher_anchor_final_objectness_weight=args.teacher_anchor_final_objectness_weight,
                teacher_anchor_bootstrap_epochs=args.teacher_anchor_bootstrap_epochs,
                teacher_anchor_decay_epochs=args.teacher_anchor_decay_epochs,
                enable_teacher_distillation=args.teacher_distillation,
                grad_accum_steps=args.grad_accum_steps,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                seed=args.seed,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                teacher_provider_config=teacher_provider_config,
            )
        )
        return
    if args.command == "train-openlane":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for train-openlane")
        config = _resolve_config(args)
        print(
            fit_openlane(
                dataroot=args.dataset_root,
                artifact_dir=args.artifact_dir / "openlane",
                config=config,
                train_split="training",
                val_split="validation",
                subset=args.subset,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_accum_steps=args.grad_accum_steps,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                seed=args.seed,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                augmentation_mode=args.augmentation_mode,
            )
        )
        return
    if args.command == "train-joint-public":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for train-joint-public")
        if args.lane_dataset_root is None:
            raise ValueError("--lane-dataset-root is required for train-joint-public")
        config = _resolve_config(args)
        teacher_provider_config = _resolve_teacher_provider_config(args)
        print(
            fit_joint_public(
                nuscenes_root=args.dataset_root,
                openlane_root=args.lane_dataset_root,
                artifact_dir=args.artifact_dir / "joint_public",
                config=config,
                nuscenes_version=args.version,
                nuscenes_train_split=args.train_split,
                nuscenes_val_split=_resolve_nuscenes_eval_split(args.version, args.split),
                openlane_subset=args.subset,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_accum_steps=args.grad_accum_steps,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                seed=args.seed,
                init_checkpoint=args.init_checkpoint,
                teacher_provider_config=teacher_provider_config,
                optimizer_schedule=(
                    args.optimizer_schedule if args.optimizer_schedule is not None else "constant"
                ),
                grad_clip_norm=5.0 if args.grad_clip_norm is None else args.grad_clip_norm,
                keep_best_checkpoint=args.keep_best_checkpoint,
                early_stop_patience=(
                    args.early_stop_patience if args.early_stop_patience is not None else 6
                ),
                early_stop_min_delta=(
                    args.early_stop_min_delta if args.early_stop_min_delta is not None else 0.01
                ),
                early_stop_min_epochs=(
                    args.early_stop_min_epochs if args.early_stop_min_epochs is not None else 6
                ),
                augmentation_mode=args.augmentation_mode,
                loss_mode=args.loss_mode,
                hard_negative_ratio=args.hard_negative_ratio,
                hard_negative_cap=args.hard_negative_cap,
                teacher_anchor_class_weight=args.teacher_anchor_class_weight,
                teacher_anchor_quality_class_weight=args.teacher_anchor_quality_class_weight,
                teacher_anchor_objectness_weight=args.teacher_anchor_objectness_weight,
                teacher_region_objectness_weight=args.teacher_region_objectness_weight,
                teacher_region_class_weight=args.teacher_region_class_weight,
                teacher_region_radius_m=args.teacher_region_radius_m,
                enable_teacher_distillation=args.teacher_distillation,
                lane_batch_multiplier=args.lane_batch_multiplier,
                official_eval_every_epochs=args.official_eval_every_epochs,
                official_eval_score_threshold=args.official_eval_score_threshold,
                official_eval_top_k=args.official_eval_top_k,
                openlane_repo_root=args.openlane_repo_root,
            )
        )
        return
    if args.command == "overfit-nuscenes":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for overfit-nuscenes")
        config = _resolve_config(args)
        teacher_provider_config = _resolve_teacher_provider_config(args)
        print(
            run_nuscenes_overfit_gate(
                dataroot=args.dataset_root,
                artifact_dir=args.artifact_dir,
                config=config,
                version=args.version,
                split=args.train_split or "mini_train",
                subset_size=args.subset_size,
                scene_name=args.scene_name,
                epochs=args.epochs,
                max_train_steps=args.max_train_steps or 1024,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_accum_steps=args.grad_accum_steps,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                seed=args.seed,
                init_checkpoint=args.init_checkpoint,
                teacher_provider_config=teacher_provider_config,
                optimizer_schedule=(
                    args.optimizer_schedule if args.optimizer_schedule is not None else "constant"
                ),
                grad_clip_norm=5.0 if args.grad_clip_norm is None else args.grad_clip_norm,
                keep_best_checkpoint=args.keep_best_checkpoint,
                early_stop_patience=(
                    args.early_stop_patience if args.early_stop_patience is not None else 8
                ),
                early_stop_min_delta=(
                    args.early_stop_min_delta if args.early_stop_min_delta is not None else 0.02
                ),
                early_stop_min_epochs=(
                    args.early_stop_min_epochs if args.early_stop_min_epochs is not None else 16
                ),
                augmentation_mode=args.augmentation_mode,
                loss_mode=args.loss_mode,
                hard_negative_ratio=args.hard_negative_ratio,
                hard_negative_cap=args.hard_negative_cap,
                teacher_anchor_class_weight=args.teacher_anchor_class_weight,
                teacher_anchor_quality_class_weight=args.teacher_anchor_quality_class_weight,
                teacher_anchor_objectness_weight=args.teacher_anchor_objectness_weight,
                teacher_region_objectness_weight=args.teacher_region_objectness_weight,
                teacher_region_class_weight=args.teacher_region_class_weight,
                teacher_region_radius_m=args.teacher_region_radius_m,
                teacher_anchor_final_class_weight=args.teacher_anchor_final_class_weight,
                teacher_anchor_final_objectness_weight=args.teacher_anchor_final_objectness_weight,
                teacher_anchor_bootstrap_epochs=args.teacher_anchor_bootstrap_epochs,
                teacher_anchor_decay_epochs=args.teacher_anchor_decay_epochs,
                enable_teacher_distillation=args.teacher_distillation,
                score_threshold_candidates=tuple(
                    args.score_threshold_candidates
                    if args.score_threshold_candidates is not None
                    else (0.05, 0.15, 0.25)
                ),
                top_k_candidates=tuple(
                    args.top_k_candidates
                    if args.top_k_candidates is not None
                    else (32, 64, 112)
                ),
            )
        )
        return
    if args.command == "cache-teacher-nuscenes":
        if args.dataset_root is None or args.result_json is None:
            raise ValueError(
                "--dataset-root and --result-json are required for cache-teacher-nuscenes"
            )
        print(
            cache_nuscenes_detection_results(
                dataroot=args.dataset_root,
                version=args.version,
                result_path=args.result_json,
                cache_dir=args.teacher_cache_dir or (args.artifact_dir / "teacher_cache"),
                top_k=args.top_k,
            )
        )
        return
    if args.command == "check-openpcdet-env":
        print(check_openpcdet_environment(args.openpcdet_repo_root))
        return
    if args.command == "audit-teacher-cache-nuscenes":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for audit-teacher-cache-nuscenes")
        if args.teacher_cache_dir is None:
            raise ValueError("--teacher-cache-dir is required for audit-teacher-cache-nuscenes")
        print(
            audit_nuscenes_teacher_cache(
                dataroot=args.dataset_root,
                version=args.version,
                split=args.split or _resolve_nuscenes_eval_split(args.version, args.split),
                cache_dir=args.teacher_cache_dir,
                max_samples=args.max_audit_samples,
                output_dir=args.output_dir,
            )
        )
        return
    if args.command == "export-nuscenes":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for export-nuscenes")
        model = _model_for_export(_resolve_config(args), args.checkpoint)
        split = _resolve_nuscenes_eval_split(args.version, args.split)
        teacher_provider_config = _resolve_teacher_provider_config(args)
        print(
            {
                "result_path": str(
                    export_nuscenes_predictions(
                        model=model,
                        dataroot=args.dataset_root,
                        version=args.version,
                        split=split,
                        output_path=args.output_path,
                        score_threshold=args.score_threshold,
                        top_k=args.top_k,
                        device=args.device,
                        teacher_provider_config=teacher_provider_config,
                    )
                )
            }
        )
        return
    if args.command == "eval-nuscenes":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for eval-nuscenes")
        split = _resolve_nuscenes_eval_split(args.version, args.split)
        result_path = args.result_json if args.result_json is not None else args.output_path
        print(
            evaluate_nuscenes_predictions(
                dataroot=args.dataset_root,
                version=args.version,
                split=split,
                result_path=result_path,
                output_dir=args.output_dir / "nuscenes",
            )
        )
        return
    if args.command == "export-openlane":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for export-openlane")
        config = _resolve_config(args).model_copy(update={"views": 1})
        model = _model_for_export(config, args.checkpoint)
        print(
            {
                "pred_dir": str(
                    export_openlane_predictions(
                        model=model,
                        dataroot=args.dataset_root,
                        output_dir=args.output_dir / "openlane_predictions",
                        split="validation",
                        subset=args.subset,
                        score_threshold=args.score_threshold,
                        max_lanes=args.top_k,
                        device=args.device,
                    )
                )
            }
        )
        return
    if args.command == "eval-openlane":
        if args.dataset_root is None or args.test_list is None:
            raise ValueError("--dataset-root and --test-list are required for eval-openlane")
        print(
            evaluate_openlane_predictions(
                openlane_repo_root=args.openlane_repo_root,
                dataset_dir=args.dataset_root / args.subset,
                pred_dir=args.output_dir / "openlane_predictions",
                test_list=args.test_list,
            )
        )
        return
    if args.command == "research-loop":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for research-loop")
        teacher_provider_config = _resolve_teacher_provider_config(args)
        print(
            run_bounded_research_loop(
                dataroot=args.dataset_root,
                artifact_dir=args.artifact_dir,
                device=args.device,
                max_experiments=args.max_experiments,
                teacher_provider_config=teacher_provider_config,
            )
        )
        return
    if args.command == "research-supervisor":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for research-supervisor")
        teacher_provider_config = _resolve_teacher_provider_config(args)
        print(
            run_research_supervisor(
                dataroot=args.dataset_root,
                artifact_dir=args.artifact_dir,
                device=args.device,
                max_experiments=args.max_experiments,
                teacher_provider_config=teacher_provider_config,
                max_invocations=args.max_invocations,
                sleep_seconds=args.sleep_seconds,
                wait_poll_seconds=args.wait_poll_seconds,
                git_publish=args.git_publish,
                git_remote=args.git_remote,
                git_branch=args.git_branch,
            )
        )
        return
    if args.command == "maintenance-once":
        print(run_maintenance_once(artifact_dir=args.artifact_dir))
        return
    if args.command == "maintenance-supervisor":
        print(
            run_maintenance_supervisor(
                artifact_dir=args.artifact_dir,
                interval_hours=args.interval_hours,
            )
        )
        return
    raise ValueError(f"unsupported command: {args.command}")
