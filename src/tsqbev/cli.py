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

from rich import print

from tsqbev.checkpoints import load_model_from_checkpoint
from tsqbev.config import ModelConfig
from tsqbev.data_checks import check_nuscenes_root, check_openlane_root
from tsqbev.eval_nuscenes import evaluate_nuscenes_predictions, export_nuscenes_predictions
from tsqbev.eval_openlane import evaluate_openlane_predictions, export_openlane_predictions
from tsqbev.export import export_core_to_onnx
from tsqbev.latency import LatencyPredictor, features_from_config
from tsqbev.model import TSQBEVModel
from tsqbev.research import run_bounded_research_loop
from tsqbev.runtime import benchmark_forward, run_eval_step, run_train_step
from tsqbev.synthetic import make_synthetic_batch
from tsqbev.teacher_backends import TeacherProviderConfig
from tsqbev.train import fit_nuscenes, fit_openlane
from tsqbev.trt import run_trt_benchmark


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
    if args.teacher_seed_mode is not None:
        updates["teacher_seed_mode"] = args.teacher_seed_mode
    return config.model_copy(update=updates)


def _resolve_nuscenes_eval_split(version: str, split: str | None) -> str:
    if split is not None:
        return split
    return "mini_val" if version == "v1.0-mini" else "val"


def _resolve_teacher_provider_config(args: argparse.Namespace) -> TeacherProviderConfig | None:
    if args.teacher_kind is None:
        return None
    return TeacherProviderConfig(
        kind=args.teacher_kind,
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
            "export-nuscenes",
            "eval-nuscenes",
            "export-openlane",
            "eval-openlane",
            "research-loop",
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/eval/predictions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=("default", "small", "rtx5000-nuscenes", "rtx5000-nuscenes-teacher"),
        default="default",
    )
    parser.add_argument(
        "--image-backbone",
        choices=("tiny", "mobilenet_v3_large", "efficientnet_b0"),
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
    parser.add_argument(
        "--teacher-seed-mode",
        choices=("off", "replace_lidar"),
        default=None,
    )
    parser.add_argument("--openlane-repo-root", type=Path, default=Path("/tmp/OpenLane"))
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--subset", type=str, default="lane3d_300")
    parser.add_argument("--test-list", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-experiments", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument(
        "--teacher-kind",
        choices=("cache", "openpcdet-centerpoint-pointpillar", "openpcdet-centerpoint-voxel"),
        default=None,
    )
    parser.add_argument("--teacher-cache-dir", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
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
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_accum_steps=args.grad_accum_steps,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
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
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
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
        print(
            evaluate_nuscenes_predictions(
                dataroot=args.dataset_root,
                version=args.version,
                split=split,
                result_path=args.output_path,
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
    raise ValueError(f"unsupported command: {args.command}")
