from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from tsqbev import cli as cli_module
from tsqbev.cli import _resolve_config, _resolve_teacher_provider_config
from tsqbev.research import _canonical_command
from tsqbev.teacher_backends import TeacherProviderConfig


def test_resolve_teacher_provider_config_infers_cache_kind_from_cache_dir() -> None:
    args = argparse.Namespace(
        teacher_kind=None,
        teacher_cache_dir=Path("/tmp/teacher_cache"),
        teacher_checkpoint=None,
    )

    config = _resolve_teacher_provider_config(args)

    assert config is not None
    assert config.kind == "cache"
    assert config.cache_dir == "/tmp/teacher_cache"


def test_canonical_command_includes_teacher_provider_flags() -> None:
    command = _canonical_command(
        dataroot=Path("/tmp/nuscenes"),
        artifact_dir=Path("/tmp/artifacts/research_v1"),
        device="cuda",
        max_experiments=5,
        teacher_provider_config=TeacherProviderConfig(
            kind="cache",
            cache_dir="/tmp/teacher_cache",
        ),
    )

    assert "--teacher-kind cache" in command
    assert "--teacher-cache-dir /tmp/teacher_cache" in command


def test_resolve_config_supports_dinov2_teacher_preset() -> None:
    args = argparse.Namespace(
        preset="rtx5000-nuscenes-dinov2-teacher",
        image_backbone=None,
        pretrained_image_backbone=None,
        freeze_image_backbone=None,
        foundation_repo_root=None,
        activation_checkpointing=None,
        attention_backend=None,
        teacher_seed_mode=None,
        teacher_seed_selection_mode=None,
    )

    config = _resolve_config(args)

    assert config.image_backbone == "dinov2_vits14_reg"
    assert config.freeze_image_backbone is True
    assert config.ranking_mode == "quality_class_only"


def test_resolve_config_supports_dinov3_teacher_preset() -> None:
    args = argparse.Namespace(
        preset="rtx5000-nuscenes-dinov3-teacher",
        image_backbone=None,
        pretrained_image_backbone=None,
        freeze_image_backbone=None,
        foundation_repo_root=None,
        foundation_weights=None,
        activation_checkpointing=None,
        attention_backend=None,
        auto_vram_fit=None,
        sam2_repo_root=None,
        sam2_model_cfg=None,
        sam2_checkpoint=None,
        sam2_region_prior_mode=None,
        sam2_region_prior_weight=None,
        teacher_seed_mode=None,
        teacher_seed_selection_mode=None,
    )

    config = _resolve_config(args)

    assert config.image_backbone == "dinov3_vits16"
    assert config.freeze_image_backbone is True
    assert config.auto_vram_fit is True
    assert config.sam2_repo_root == "/home/achbogga/projects/sam2"
    assert config.sam2_model_cfg == "configs/sam2.1/sam2.1_hiera_b+.yaml"
    assert config.sam2_checkpoint.endswith("sam2.1_hiera_base_plus.pt")
    assert config.sam2_region_prior_mode == "proposal_boxes"


def test_resolve_config_supports_teacher_quality_plus_preset() -> None:
    args = argparse.Namespace(
        preset="rtx5000-nuscenes-teacher-quality-plus",
        image_backbone=None,
        pretrained_image_backbone=None,
        freeze_image_backbone=None,
        foundation_repo_root=None,
        foundation_weights=None,
        activation_checkpointing=None,
        attention_backend=None,
        auto_vram_fit=None,
        sam2_repo_root=None,
        sam2_model_cfg=None,
        sam2_checkpoint=None,
        sam2_region_prior_mode=None,
        sam2_region_prior_weight=None,
        teacher_seed_mode=None,
        teacher_seed_selection_mode=None,
    )

    config = _resolve_config(args)

    assert config.image_backbone == "mobilenet_v3_large"
    assert config.freeze_image_backbone is False
    assert config.teacher_seed_mode == "replace_lidar"


def test_eval_nuscenes_prefers_result_json(monkeypatch, tmp_path) -> None:
    captured: dict[str, Path] = {}

    def fake_parser() -> argparse.ArgumentParser:
        class _Parser:
            def parse_args(self_inner) -> SimpleNamespace:
                return SimpleNamespace(
                    command="eval-nuscenes",
                    dataset_root=tmp_path,
                    version="v1.0-mini",
                    split="mini_val",
                    result_json=tmp_path / "results.json",
                    output_path=tmp_path / "predictions.json",
                    output_dir=tmp_path / "artifacts",
                )

        return _Parser()  # type: ignore[return-value]

    monkeypatch.setattr(cli_module, "_make_parser", fake_parser)
    monkeypatch.setattr(
        cli_module,
        "_resolve_nuscenes_eval_split",
        lambda version, split: split or "mini_val",
    )

    def fake_eval(**kwargs):
        captured["result_path"] = Path(kwargs["result_path"])
        return {"ok": True}

    monkeypatch.setattr(cli_module, "evaluate_nuscenes_predictions", fake_eval)
    cli_module.main()
    assert captured["result_path"] == tmp_path / "results.json"


def test_train_joint_public_passes_lane_batch_multiplier(monkeypatch, tmp_path) -> None:
    captured: dict[str, float | int] = {}

    def fake_parser() -> argparse.ArgumentParser:
        class _Parser:
            def parse_args(self_inner) -> SimpleNamespace:
                return SimpleNamespace(
                    command="train-joint-public",
                    dataset_root=tmp_path / "nuscenes",
                    lane_dataset_root=tmp_path / "openlane",
                    artifact_dir=tmp_path / "artifacts",
                    preset="rtx5000-nuscenes-teacher-quality-plus",
                    image_backbone=None,
                    pretrained_image_backbone=None,
                    freeze_image_backbone=None,
                    foundation_repo_root=None,
                    activation_checkpointing=None,
                    attention_backend=None,
                    teacher_seed_mode=None,
                    teacher_seed_selection_mode=None,
                    teacher_kind=None,
                    teacher_cache_dir=None,
                    teacher_checkpoint=None,
                    version="v1.0-mini",
                    train_split="mini_train",
                    split="mini_val",
                    subset="lane3d_300",
                    epochs=36,
                    lr=1e-4,
                    weight_decay=0.0,
                    grad_accum_steps=2,
                    batch_size=1,
                    num_workers=4,
                    device="cuda",
                    seed=1337,
                    init_checkpoint=None,
                    optimizer_schedule=None,
                    grad_clip_norm=None,
                    keep_best_checkpoint=True,
                    early_stop_patience=None,
                    early_stop_min_delta=None,
                    early_stop_min_epochs=None,
                    augmentation_mode="off",
                    loss_mode="quality_focal",
                    hard_negative_ratio=3,
                    hard_negative_cap=96,
                    teacher_anchor_class_weight=0.5,
                    teacher_anchor_quality_class_weight=0.45,
                    teacher_anchor_objectness_weight=0.5,
                    teacher_region_objectness_weight=0.12,
                    teacher_region_class_weight=0.12,
                    teacher_region_radius_m=4.0,
                    teacher_distillation=True,
                    lane_batch_multiplier=0.75,
                    official_eval_every_epochs=5,
                    official_eval_score_threshold=0.2,
                    official_eval_top_k=40,
                    openlane_repo_root=tmp_path / "OpenLane",
                )

        return _Parser()  # type: ignore[return-value]

    monkeypatch.setattr(cli_module, "_make_parser", fake_parser)
    monkeypatch.setattr(
        cli_module,
        "_resolve_nuscenes_eval_split",
        lambda version, split: split or "mini_val",
    )

    def fake_fit_joint_public(**kwargs):
        captured["lane_batch_multiplier"] = float(kwargs["lane_batch_multiplier"])
        captured["official_eval_every_epochs"] = int(kwargs["official_eval_every_epochs"])
        captured["official_eval_top_k"] = int(kwargs["official_eval_top_k"])
        return {"ok": True}

    monkeypatch.setattr(cli_module, "fit_joint_public", fake_fit_joint_public)
    cli_module.main()
    assert captured["lane_batch_multiplier"] == 0.75


def test_harness_search_passes_provider_and_iterations(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_parser() -> argparse.ArgumentParser:
        class _Parser:
            def parse_args(self_inner) -> SimpleNamespace:
                return SimpleNamespace(
                    command="harness-search",
                    artifact_dir=tmp_path / "artifacts" / "harness_v2",
                    proposal_path=tmp_path / "proposal.md",
                    iterations=4,
                    harness_provider="heuristic",
                    budget_chars=12000,
                )

        return _Parser()  # type: ignore[return-value]

    monkeypatch.setattr(cli_module, "_make_parser", fake_parser)

    def fake_search(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli_module, "run_harness_search", fake_search)
    cli_module.main()


def test_codex_loop_dispatches_to_runner(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_parser() -> argparse.ArgumentParser:
        class _Parser:
            def parse_args(self_inner) -> SimpleNamespace:
                return SimpleNamespace(
                    command="codex-loop",
                    dataset_root=tmp_path / "nuscenes",
                    artifact_dir=tmp_path / "artifacts" / "codex_loop",
                    proposal_path=tmp_path / "proposal.md",
                    device="cuda",
                    max_experiments=3,
                    teacher_kind=None,
                    teacher_cache_dir=None,
                    teacher_checkpoint=None,
                    max_cycles=2,
                    iterations=4,
                    sleep_seconds=7,
                    wait_poll_seconds=11,
                    git_publish=False,
                    git_remote="origin",
                    git_branch="main",
                )

        return _Parser()  # type: ignore[return-value]

    monkeypatch.setattr(cli_module, "_make_parser", fake_parser)

    def fake_loop(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli_module, "run_codex_loop", fake_loop)
    cli_module.main()

    assert captured["dataroot"] == tmp_path / "nuscenes"
    assert captured["artifact_dir"] == tmp_path / "artifacts" / "codex_loop"
    assert captured["harness_root"] == tmp_path / "artifacts" / "codex_loop" / "harness_v2"
    assert captured["max_cycles"] == 2
    assert captured["search_iterations"] == 4


def test_run_bevdet_public_student_dispatches(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_parser() -> argparse.ArgumentParser:
        class _Parser:
            def parse_args(self_inner) -> SimpleNamespace:
                return SimpleNamespace(
                    command="run-bevdet-public-student",
                    dataset_root=tmp_path / "nuscenes",
                    artifact_dir=tmp_path / "artifacts",
                    bevdet_repo_root=tmp_path / "BEVDet",
                    bevdet_config_rel="configs/bevdet/bevdet-r50-cbgs.py",
                    bevdet_docker_image_tag="tsqbev-bevdet-official:latest",
                    version="v1.0-mini",
                    epochs=1,
                    batch_size=1,
                    num_workers=2,
                    init_checkpoint=None,
                )

        return _Parser()  # type: ignore[return-value]

    monkeypatch.setattr(cli_module, "_make_parser", fake_parser)

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli_module, "run_bevdet_public_student", fake_run)
    cli_module.main()

    assert captured["config_relpath"] == "configs/bevdet/bevdet-r50-cbgs.py"
    assert captured["version"] == "v1.0-mini"
    assert captured["epochs"] == 1
    assert captured["samples_per_gpu"] == 1
    assert captured["workers_per_gpu"] == 2


def test_train_nuscenes_passes_official_eval_controls(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_parser() -> argparse.ArgumentParser:
        class _Parser:
            def parse_args(self_inner) -> SimpleNamespace:
                return SimpleNamespace(
                    command="train-nuscenes",
                    dataset_root=tmp_path / "nuscenes",
                    artifact_dir=tmp_path / "artifacts",
                    preset="rtx5000-nuscenes-dinov3-teacher",
                    image_backbone=None,
                    pretrained_image_backbone=None,
                    freeze_image_backbone=None,
                    foundation_repo_root=None,
                    foundation_weights=None,
                    activation_checkpointing=None,
                    auto_vram_fit=None,
                    attention_backend=None,
                    sam2_repo_root=None,
                    sam2_model_cfg=None,
                    sam2_checkpoint=None,
                    sam2_region_prior_mode=None,
                    sam2_region_prior_weight=None,
                    teacher_seed_mode=None,
                    teacher_seed_selection_mode=None,
                    version="v1.0-mini",
                    train_split="mini_train",
                    split="mini_val",
                    epochs=36,
                    max_train_steps=None,
                    init_checkpoint=None,
                    lr=1e-4,
                    weight_decay=0.0,
                    optimizer_schedule=None,
                    grad_clip_norm=None,
                    keep_best_checkpoint=True,
                    early_stop_patience=6,
                    early_stop_min_delta=0.02,
                    early_stop_min_epochs=10,
                    official_eval_every_epochs=5,
                    official_eval_score_threshold=0.2,
                    official_eval_top_k=40,
                    augmentation_mode="off",
                    loss_mode="quality_focal",
                    hard_negative_ratio=3,
                    hard_negative_cap=96,
                    teacher_anchor_class_weight=0.5,
                    teacher_anchor_quality_class_weight=0.45,
                    teacher_anchor_objectness_weight=0.5,
                    teacher_region_objectness_weight=0.12,
                    teacher_region_class_weight=0.12,
                    teacher_region_radius_m=4.0,
                    teacher_anchor_final_class_weight=None,
                    teacher_anchor_final_objectness_weight=None,
                    teacher_anchor_bootstrap_epochs=0,
                    teacher_anchor_decay_epochs=0,
                    teacher_distillation=True,
                    grad_accum_steps=8,
                    batch_size=1,
                    num_workers=4,
                    device="cuda",
                    seed=1337,
                    max_train_samples=None,
                    max_val_samples=None,
                    teacher_kind="cache",
                    teacher_cache_dir=tmp_path / "teacher_cache",
                    teacher_checkpoint=None,
                )

        return _Parser()  # type: ignore[return-value]

    monkeypatch.setattr(cli_module, "_make_parser", fake_parser)
    monkeypatch.setattr(
        cli_module,
        "_resolve_nuscenes_eval_split",
        lambda version, split: split or "mini_val",
    )

    def fake_fit_nuscenes(**kwargs):
        captured["official_eval_every_epochs"] = kwargs["official_eval_every_epochs"]
        captured["official_eval_score_threshold"] = kwargs["official_eval_score_threshold"]
        captured["official_eval_top_k"] = kwargs["official_eval_top_k"]
        return {"ok": True}

    monkeypatch.setattr(cli_module, "fit_nuscenes", fake_fit_nuscenes)
    cli_module.main()
    assert captured["official_eval_every_epochs"] == 5
    assert captured["official_eval_score_threshold"] == 0.2
    assert captured["official_eval_top_k"] == 40


def test_check_sam2_assets_validates_local_paths(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_parser() -> argparse.ArgumentParser:
        class _Parser:
            def parse_args(self_inner) -> SimpleNamespace:
                return SimpleNamespace(
                    command="check-sam2-assets",
                    dataset_root=tmp_path / "nuscenes",
                    lane_dataset_root=tmp_path / "openlane",
                    artifact_dir=tmp_path / "artifacts",
                    preset="rtx5000-nuscenes-dinov3-teacher",
                    image_backbone=None,
                    pretrained_image_backbone=None,
                    freeze_image_backbone=None,
                    foundation_repo_root=None,
                    foundation_weights=None,
                    activation_checkpointing=None,
                    auto_vram_fit=None,
                    attention_backend=None,
                    sam2_repo_root=None,
                    sam2_model_cfg=None,
                    sam2_checkpoint=None,
                    sam2_region_prior_mode=None,
                    sam2_region_prior_weight=None,
                    teacher_seed_mode=None,
                    teacher_seed_selection_mode=None,
                    version="v1.0-mini",
                    train_split="mini_train",
                    split="mini_val",
                    subset="lane3d_300",
                    epochs=36,
                    max_train_steps=None,
                    lr=1e-4,
                    weight_decay=0.0,
                    augmentation_mode="off",
                    optimizer_schedule=None,
                    grad_clip_norm=None,
                    keep_best_checkpoint=True,
                    teacher_distillation=True,
                    loss_mode="quality_focal",
                    hard_negative_ratio=3,
                    hard_negative_cap=96,
                    teacher_anchor_class_weight=0.5,
                    teacher_anchor_quality_class_weight=0.45,
                    teacher_anchor_objectness_weight=0.5,
                    teacher_region_objectness_weight=0.12,
                    teacher_region_class_weight=0.12,
                    teacher_region_radius_m=4.0,
                    lane_batch_multiplier=1.0,
                    official_eval_every_epochs=5,
                    official_eval_score_threshold=0.2,
                    official_eval_top_k=40,
                    teacher_anchor_final_class_weight=None,
                    teacher_anchor_final_objectness_weight=None,
                    teacher_anchor_bootstrap_epochs=0,
                    teacher_anchor_decay_epochs=0,
                    early_stop_patience=6,
                    early_stop_min_delta=0.02,
                    early_stop_min_epochs=10,
                    grad_accum_steps=8,
                    batch_size=1,
                    num_workers=4,
                    device="cpu",
                    seed=1337,
                    max_train_samples=None,
                    max_val_samples=None,
                    teacher_kind=None,
                    teacher_cache_dir=None,
                    teacher_checkpoint=None,
                    score_threshold=0.25,
                    score_threshold_candidates=None,
                    top_k=300,
                    top_k_candidates=None,
                    subset_size=32,
                    max_audit_samples=None,
                    scene_name=None,
                    max_experiments=5,
                    max_invocations=None,
                    sleep_seconds=30,
                    wait_poll_seconds=20,
                    interval_hours=24,
                    git_publish=True,
                    git_remote="origin",
                    git_branch=None,
                    openlane_repo_root=tmp_path / "OpenLane",
                    bevfusion_repo_root=tmp_path / "bevfusion",
                    docker_image_tag="tsqbev-bevfusion-official:latest",
                    num_gpus=1,
                    output_path=tmp_path / "predictions.json",
                    output_dir=tmp_path / "artifacts",
                    result_json=None,
                    query=None,
                    limit=8,
                    archive_key=None,
                    extract_openlanev2=True,
                    overwrite_download=False,
                    include_lane3d_1000=False,
                    include_scene=False,
                    include_cipo=False,
                    force_reextract=False,
                    max_openlane_train_samples=None,
                    max_openlane_val_samples=None,
                )

        return _Parser()  # type: ignore[return-value]

    monkeypatch.setattr(cli_module, "_make_parser", fake_parser)

    def fake_validate(config, device=None):
        captured["repo_root"] = config.sam2_repo_root
        captured["checkpoint"] = config.sam2_checkpoint
        captured["device"] = device
        return {"status": "ready"}

    monkeypatch.setattr(cli_module, "validate_local_sam2_assets", fake_validate)
    cli_module.main()
    assert captured["repo_root"] == "/home/achbogga/projects/sam2"
    assert str(captured["checkpoint"]).endswith("sam2.1_hiera_base_plus.pt")
    assert captured["device"] == "cpu"


def test_make_parser_defaults_openlane_repo_root_to_local_checkout() -> None:
    parser = cli_module._make_parser()
    action = next(action for action in parser._actions if action.dest == "openlane_repo_root")
    assert action.default == Path("/home/achbogga/projects/OpenLane")


def test_maintenance_supervisor_dispatches_interval(monkeypatch, tmp_path) -> None:
    captured: dict[str, int | str] = {}

    def fake_parser() -> argparse.ArgumentParser:
        class _Parser:
            def parse_args(self_inner) -> SimpleNamespace:
                return SimpleNamespace(
                    command="maintenance-supervisor",
                    artifact_dir=tmp_path / "maintenance",
                    interval_hours=24,
                )

        return _Parser()  # type: ignore[return-value]

    monkeypatch.setattr(cli_module, "_make_parser", fake_parser)

    def fake_run_maintenance_supervisor(**kwargs):
        captured["interval_hours"] = int(kwargs["interval_hours"])
        captured["artifact_dir"] = str(kwargs["artifact_dir"])
        return {"ok": True}

    monkeypatch.setattr(cli_module, "run_maintenance_supervisor", fake_run_maintenance_supervisor)
    cli_module.main()
    assert captured["interval_hours"] == 24
