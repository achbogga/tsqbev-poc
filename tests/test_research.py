from __future__ import annotations

import json
from pathlib import Path

from tsqbev import research
from tsqbev.config import ModelConfig


def _fake_eval(nds: float, mean_ap: float, car_ap_4m: float = 0.0) -> dict[str, object]:
    return {
        "mean_ap": mean_ap,
        "nd_score": nds,
        "label_aps": {
            "car": {"0.5": 0.0, "1.0": 0.0, "2.0": 0.0, "4.0": car_ap_4m},
            "truck": {"0.5": 0.0, "1.0": 0.0, "2.0": 0.0, "4.0": 0.0},
        },
        "tp_errors": {"trans_err": 0.95 if nds >= 0.03 else 1.05},
    }


def _fake_selected_record(
    *,
    recipe: str,
    nds: float,
    mean_ap: float,
    val_total: float,
    boxes_mean: float,
    root_cause_verdict: str = "incremental_progress",
    ranking_mode: str = "class_times_objectness",
    lidar_share: float = 1.0,
    proposal_share: float = 0.0,
) -> dict[str, object]:
    return {
        "recipe": recipe,
        "evaluation": _fake_eval(nds, mean_ap, car_ap_4m=0.1),
        "val": {"total": val_total},
        "prediction_geometry": {"boxes_per_sample_mean": boxes_mean},
        "root_cause_verdict": root_cause_verdict,
        "use_teacher_provider": True,
        "config": {"ranking_mode": ranking_mode},
        "source_mix": {
            "lidar": lidar_share,
            "proposal": proposal_share,
            "global": max(0.0, 1.0 - lidar_share - proposal_share),
        },
        "teacher_anchor_quality_class_weight": 0.35,
    }


def _fake_calibration_selected(
    kwargs: dict[str, object],
    evals: dict[str, dict[str, object]],
) -> dict[str, object]:
    output_dir = Path(str(kwargs["output_dir"]))
    run_name = output_dir.parent.name
    return {
        "score_threshold": 0.05,
        "top_k": 32,
        "prediction_path": str(output_dir / f"{run_name}.json"),
        "evaluation": evals[run_name],
    }


def test_run_bounded_research_loop_writes_autoresearch_ledgers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    dataset_root = tmp_path / "nuscenes"
    dataset_root.mkdir()

    vals = {
        "mini_balanced_mbv3_frozen": 21.0,
        "mini_propheavy_mbv3_frozen": 22.0,
        "mini_propheavy_effb0_frozen": 23.0,
        "mini_propheavy_effb0_frozen_quality_focal": 21.25,
        "mini_propheavy_effb0_frozen_moderate_aug": 22.5,
        "mini_propheavy_effb0_frozen_query_boost": 20.0,
    }
    evals = {
        "mini_balanced_mbv3_frozen": _fake_eval(0.0, 0.0),
        "mini_propheavy_mbv3_frozen": _fake_eval(0.0, 0.0),
        "mini_propheavy_effb0_frozen": _fake_eval(0.01, 0.0),
        "mini_propheavy_effb0_frozen_quality_focal": _fake_eval(0.02, 0.002),
        "mini_propheavy_effb0_frozen_moderate_aug": _fake_eval(0.015, 0.001),
        "mini_propheavy_effb0_frozen_query_boost": _fake_eval(0.03, 0.01, car_ap_4m=0.02),
    }

    monkeypatch.setattr(research, "ensure_research_loop_enabled", lambda: None)

    def fake_fit_nuscenes(**kwargs: object) -> dict[str, object]:
        run_dir = Path(str(kwargs["artifact_dir"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "checkpoint_last.pt"
        checkpoint.write_text("checkpoint")
        run_name = run_dir.name
        return {
            "checkpoint_path": str(checkpoint),
            "selected_checkpoint_path": str(checkpoint),
            "best_checkpoint_path": str(checkpoint),
            "selected_epoch": 4,
            "best_epoch": 4,
            "selected_train": {"total": 39.0},
            "selected_val": {"total": vals[run_name]},
            "best_val": {"total": vals[run_name]},
            "last_train": {"total": 40.0},
            "last_val": {"total": vals[run_name]},
            "train_samples": 16,
            "val_samples": 8,
        }

    monkeypatch.setattr(research, "fit_nuscenes", fake_fit_nuscenes)
    monkeypatch.setattr(
        research,
        "benchmark_forward",
        lambda *args, **kwargs: {"mean_ms": 21.5, "p95_ms": 22.0},
    )
    monkeypatch.setattr(
        research,
        "load_model_from_checkpoint",
        lambda *args, **kwargs: (object(), {}),
    )

    monkeypatch.setattr(
        research,
        "export_and_evaluate_nuscenes_grid",
        lambda **kwargs: {
            "selected": _fake_calibration_selected(kwargs, evals),
            "candidates": [],
        },
    )
    monkeypatch.setattr(
        research,
        "prediction_geometry_diagnostics",
        lambda *args, **kwargs: {
            "boxes_per_sample_mean": 12.0,
            "boxes_per_sample_p95": 14.0,
            "boxes_per_sample_max": 16.0,
            "ego_translation_norm_mean": 18.0,
            "ego_translation_norm_p95": 24.0,
            "ego_translation_norm_p99": 30.0,
            "ego_translation_norm_max": 36.0,
        },
    )
    monkeypatch.setattr(
        research,
        "_measure_source_mix",
        lambda *args, **kwargs: {
            "average": {"lidar": 0.33, "proposal": 0.50, "global": 0.17},
            "per_batch": [{"lidar": 0.33, "proposal": 0.50, "global": 0.17}] * 8,
            "batches_measured": 8,
        },
    )
    monkeypatch.setattr(research, "_current_git_sha", lambda: "deadbee")

    jsonl_snapshots: list[int] = []
    tsv_snapshots: list[int] = []
    original_write_jsonl = research._write_jsonl
    original_write_results_tsv = research._write_results_tsv

    def tracking_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        jsonl_snapshots.append(len(rows))
        original_write_jsonl(path, rows)

    def tracking_write_results_tsv(path: Path, rows: list[dict[str, object]]) -> None:
        tsv_snapshots.append(len(rows))
        original_write_results_tsv(path, rows)

    monkeypatch.setattr(research, "_write_jsonl", tracking_write_jsonl)
    monkeypatch.setattr(research, "_write_results_tsv", tracking_write_results_tsv)

    summary = research.run_bounded_research_loop(
        dataroot=dataset_root,
        artifact_dir=artifact_dir,
        device="cpu",
        max_experiments=6,
    )

    assert summary["status"] == "completed"
    assert summary["selected_recipe"] == "mini_propheavy_effb0_frozen_query_boost"
    assert summary["scale_gate_verdict"]["authorized"] is False
    assert any(
        "CenterPoint-PointPillar" in step for step in summary["recommended_next_steps"]
    )

    results_path = artifact_dir / "research_loop" / "results.jsonl"
    tsv_path = artifact_dir / "research_loop" / "results.tsv"
    assert results_path.exists()
    assert tsv_path.exists()

    records = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert len(records) == 6
    promoted = [record for record in records if record["final_decision"] == "promote"]
    assert len(promoted) == 1
    assert promoted[0]["recipe"] == "mini_propheavy_effb0_frozen_query_boost"
    assert promoted[0]["final_rank"] == 1

    tsv_lines = tsv_path.read_text().splitlines()
    assert tsv_lines[0].startswith("run_id\trecipe\tstage")
    assert len(tsv_lines) == 7
    assert jsonl_snapshots[:6] == [1, 2, 3, 4, 5, 6]
    assert tsv_snapshots[:6] == [1, 2, 3, 4, 5, 6]
    assert jsonl_snapshots[-1] == 6
    assert tsv_snapshots[-1] == 6

    manifest_path = (
        artifact_dir
        / "research_loop"
        / "mini_propheavy_effb0_frozen_query_boost"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["git_sha"] == "deadbee"
    assert manifest["recipe"]["recipe"] == "mini_propheavy_effb0_frozen_query_boost"


def test_run_bounded_research_loop_stops_on_catastrophic_training_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    dataset_root = tmp_path / "nuscenes"
    dataset_root.mkdir()

    monkeypatch.setattr(research, "ensure_research_loop_enabled", lambda: None)
    monkeypatch.setattr(
        research,
        "safe_build_research_brief",
        lambda *args, **kwargs: {"current_state": [], "open_blockers": []},
    )

    def fake_fit_nuscenes(**kwargs: object) -> dict[str, object]:
        run_dir = Path(str(kwargs["artifact_dir"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "checkpoint_last.pt"
        checkpoint.write_text("checkpoint")
        return {
            "checkpoint_path": str(checkpoint),
            "selected_checkpoint_path": str(checkpoint),
            "best_checkpoint_path": str(checkpoint),
            "selected_epoch": 5,
            "best_epoch": 5,
            "epochs": 5,
            "selected_train": {"total": 18.0},
            "selected_val": {"total": 16.0},
            "last_train": {"total": 18.0},
            "last_val": {"total": 16.0},
            "train_samples": 16,
            "val_samples": 8,
            "early_stop_triggered": True,
            "early_stop_reason": (
                "catastrophic official-eval failure: epoch=5 nds=0.0000 map=0.0000 "
                "sanity_ok=0 max_box_size_m=1200.00 score_mean=0.9999"
            ),
            "latest_official_eval": {
                "nuscenes_nds": 0.0,
                "nuscenes_map": 0.0,
                "nuscenes_export_sanity_ok": 0.0,
                "nuscenes_boxes_per_sample": 40.0,
                "nuscenes_ego_translation_norm_p99": 25.0,
                "nuscenes_ego_translation_norm_max": 30.0,
                "nuscenes_max_box_size_m": 1200.0,
                "nuscenes_score_mean": 0.9999,
            },
        }

    monkeypatch.setattr(research, "fit_nuscenes", fake_fit_nuscenes)
    monkeypatch.setattr(
        research,
        "benchmark_forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not benchmark")),
    )
    monkeypatch.setattr(
        research,
        "export_and_evaluate_nuscenes_grid",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not export/eval")),
    )
    monkeypatch.setattr(research, "_current_git_sha", lambda: "deadbee")

    summary = research.run_bounded_research_loop(
        dataroot=dataset_root,
        artifact_dir=artifact_dir,
        device="cpu",
        max_experiments=1,
        proposal={
            "proposal_id": "catastrophe-test",
            "objective": "test",
            "launch_tags": ["dino_v3"],
            "exploitation_tags": ["dino_v3"],
            "suppress_tags": [],
            "rationale": [],
            "kill_conditions": [],
            "force_tags_only": True,
        },
    )

    assert summary["status"] == "failed"
    results_path = artifact_dir / "research_loop" / "results.jsonl"
    records = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "catastrophic_stop"
    assert records[0]["root_cause_verdict"] == "catastrophic_geometry_failure"


def test_initial_recipes_can_carry_forward_previous_incumbent(tmp_path: Path) -> None:
    research_root = tmp_path / "repo"
    research_root.mkdir(parents=True)
    original_repo_root = research.REPO_ROOT
    research.REPO_ROOT = research_root
    try:
        artifact_root = tmp_path / "research_loop"
        artifact_root.mkdir(parents=True)
        summary_path = artifact_root / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "selected_record": {
                        "recipe": "mini_propheavy_effb0_frozen",
                        "config": ModelConfig.rtx5000_nuscenes_baseline().model_dump(),
                        "batch_size": 2,
                        "grad_accum_steps": 2,
                        "lr": 2e-4,
                        "epochs": 6,
                        "num_workers": 4,
                        "score_threshold": 0.05,
                        "top_k": 300,
                    }
                }
            )
        )

        recipes = research._initial_recipes(artifact_root)

        assert recipes[0].name == "carryover_mini_propheavy_effb0_frozen"
        assert recipes[0].stage == "baseline"
        assert recipes[0].parent_recipe == "mini_propheavy_effb0_frozen"
        assert recipes[0].use_teacher_provider is False
    finally:
        research.REPO_ROOT = original_repo_root


def test_initial_recipes_insert_teacher_kd_when_teacher_is_available(tmp_path: Path) -> None:
    research_root = tmp_path / "repo"
    research_root.mkdir(parents=True)
    original_repo_root = research.REPO_ROOT
    research.REPO_ROOT = research_root
    try:
        artifact_root = tmp_path / "research_loop"
        artifact_root.mkdir(parents=True)
        summary_path = artifact_root / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "selected_record": {
                        "recipe": "mini_propheavy_effb0_frozen",
                        "config": ModelConfig.rtx5000_nuscenes_baseline().model_dump(),
                        "batch_size": 2,
                        "grad_accum_steps": 2,
                        "lr": 2e-4,
                        "epochs": 6,
                        "num_workers": 4,
                        "score_threshold": 0.05,
                        "top_k": 300,
                    }
                }
            )
        )

        recipes = research._initial_recipes(artifact_root, teacher_provider_available=True)

        assert recipes[0].name == "carryover_mini_propheavy_effb0_frozen"
        assert recipes[1].name == "carryover_mini_propheavy_effb0_frozen_teacher_seed"
        assert recipes[1].use_teacher_provider is True
        assert recipes[1].enable_teacher_distillation is False
        assert recipes[1].config.router_mode == "anchor_first"
    finally:
        research.REPO_ROOT = original_repo_root


def test_initial_recipes_obey_boss_priority_only(tmp_path: Path) -> None:
    artifact_root = tmp_path / "research_loop"
    artifact_root.mkdir(parents=True)
    summary_path = artifact_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "selected_record": {
                    "recipe": "mini_teacher_quality",
                    "config": ModelConfig.rtx5000_nuscenes_baseline()
                    .model_copy(update={"teacher_seed_mode": "replace_lidar"})
                    .model_dump(),
                    "batch_size": 2,
                    "grad_accum_steps": 1,
                    "lr": 1e-4,
                    "epochs": 6,
                    "num_workers": 4,
                    "score_threshold": 0.05,
                    "top_k": 64,
                }
            }
        )
    )

    recipes = research._initial_recipes(
        artifact_root,
        teacher_provider_available=True,
        boss_policy={
            "force_priority_only": True,
            "priority_tags": ["anchor_mix", "quality_rank"],
            "suppress_tags": ["query_boost", "lr_down"],
        },
    )

    assert [recipe.name for recipe in recipes] == ["carryover_mini_teacher_quality"]


def test_initial_recipes_launch_frontier_family_when_hard_pivot_requested(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "research_loop"
    artifact_root.mkdir()

    recipes = research._initial_recipes(
        artifact_root,
        teacher_provider_available=True,
        boss_policy={
            "force_priority_only": True,
            "priority_tags": [
                "dino_v3",
                "dino_v3_bridge",
                "sam21_offline_support",
                "world_aligned_distillation",
                "geometry_sanity",
                "official_metric_only",
            ],
            "suppress_tags": ["query_boost", "lr_down", "teacher_bag"],
        },
    )

    assert recipes[0].name == "frontier_dinov3_teacher_distill_vits16"
    assert recipes[0].config.image_backbone == "dinov3_vits16"
    assert recipes[0].use_teacher_provider is True
    assert recipes[0].enable_teacher_distillation is True
    assert recipes[0].official_eval_every_epochs == 5
    assert recipes[0].early_stop_patience == 3
    recipe_names = [recipe.name for recipe in recipes]
    assert "frontier_dinov3_teacher_distill_vits16_official_guardrail" in recipe_names
    assert "frontier_dinov3_teacher_distill_vits16_world_distill" in recipe_names
    assert "frontier_dinov3_teacher_distill_vits16_vitb16" in recipe_names
    assert "frontier_dinov3_teacher_distill_vits16_no_sam2" in recipe_names
    assert "frontier_dinov3_teacher_distill_vits16_teacher_control" in recipe_names
    assert all("carryover_" not in recipe.name for recipe in recipes)


def test_initial_recipes_launch_lightweight_bridge_family_when_requested(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "research_loop"
    artifact_root.mkdir()

    recipes = research._initial_recipes(
        artifact_root,
        teacher_provider_available=True,
        boss_policy={
            "force_priority_only": True,
            "priority_tags": [
                "lightweight_bridge",
                "gated_cross_attention",
                "teacher_side_foundation",
                "world_aligned_distillation",
                "geometry_sanity",
                "official_metric_only",
            ],
            "suppress_tags": ["query_boost", "lr_down", "teacher_bag"],
        },
    )

    assert recipes[0].name == "frontier_light_bridge_teacher_bootstrap"
    assert recipes[0].config.fusion_style == "gated_latent_cross_attn"
    assert recipes[0].config.teacher_seed_mode == "replace_lidar"
    assert recipes[0].config.router_mode == "anchor_first"
    recipe_names = [recipe.name for recipe in recipes]
    assert "frontier_light_bridge_teacher_bootstrap_official_guardrail" in recipe_names
    assert "frontier_light_bridge_teacher_bootstrap_world_distill" in recipe_names
    assert "frontier_light_bridge_teacher_bootstrap_effb0" in recipe_names
    assert "frontier_light_bridge_teacher_bootstrap_teacher_control" in recipe_names


def test_initial_recipes_launch_public_bevdet_family_when_requested(tmp_path: Path) -> None:
    artifact_root = tmp_path / "research_loop"
    artifact_root.mkdir()

    recipes = research._initial_recipes(
        artifact_root,
        teacher_provider_available=True,
        boss_policy={
            "force_priority_only": True,
            "priority_tags": [
                "public_student_replacement",
                "bevdet_public_student",
                "camera_bev_working_baseline",
                "official_box_coder",
                "bevdepth_temporal_student",
            ],
            "suppress_tags": ["query_boost", "lr_down", "teacher_bag"],
        },
    )

    assert recipes[0].name == "public_bevdet_r50_cbgs"
    assert recipes[0].execution_backend == "bevdet_official"
    assert recipes[0].external_config_relpath == "configs/bevdet/bevdet-r50-cbgs.py"
    assert recipes[1].name == "public_bevdet_r50_4d_depth_cbgs"
    assert recipes[1].execution_backend == "bevdet_official"


def test_initial_recipes_fail_loudly_when_frontier_tags_filter_everything(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "research_loop"
    artifact_root.mkdir()

    try:
        research._initial_recipes(
            artifact_root,
            teacher_provider_available=True,
            boss_policy={
                "force_priority_only": True,
                "priority_tags": ["shared_world_latent"],
                "suppress_tags": ["dino_v3", "sam21_offline_support", "world_aligned_distillation"],
            },
        )
    except RuntimeError as exc:
        assert "frontier launch tags" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected frontier launch selection to fail loudly")


def test_boss_progress_verdict_marks_breakthrough_against_previous_incumbent(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "research_vx" / "research_loop"
    artifact_root.mkdir(parents=True)
    previous = _fake_selected_record(
        recipe="prev_incumbent",
        nds=0.1507,
        mean_ap=0.1736,
        val_total=12.0182,
        boxes_mean=31.9,
    )
    current = _fake_selected_record(
        recipe="current_incumbent",
        nds=0.1712,
        mean_ap=0.1787,
        val_total=11.1095,
        boxes_mean=32.0,
    )

    verdict = research._boss_progress_verdict(
        current_record=current,
        previous_record=previous,
        artifact_root=artifact_root,
    )

    assert verdict["progress_class"] == "breakthrough"
    assert verdict["delta_vs_previous_incumbent"]["nds"] > 0.01


def test_build_exploitation_recipes_respects_boss_priority_only() -> None:
    incumbent_recipe = research.ResearchRecipe(
        name="incumbent",
        note="",
        hypothesis="",
        mutation_reason="",
        config=ModelConfig.rtx5000_nuscenes_baseline().model_copy(
            update={
                "ranking_mode": "class_times_objectness",
                "teacher_seed_mode": "replace_lidar",
            }
        ),
        use_teacher_provider=True,
        loss_mode="quality_focal",
        teacher_anchor_quality_class_weight=0.35,
        teacher_region_objectness_weight=0.1,
        teacher_region_class_weight=0.1,
    )
    incumbent_record = _fake_selected_record(
        recipe="incumbent",
        nds=0.1712,
        mean_ap=0.1787,
        val_total=11.1095,
        boxes_mean=32.0,
        lidar_share=0.8571,
        proposal_share=0.0,
    )

    recipes = research._build_exploitation_recipes(
        incumbent_recipe,
        incumbent_record,
        teacher_provider_config=None,
        remaining_budget=5,
        boss_policy={
            "force_priority_only": True,
            "priority_tags": [
                "teacher_bag",
                "quality_rank",
                "anchor_mix",
                "teacher_off_control",
            ],
            "suppress_tags": ["query_boost", "lr_down", "augmentation", "focal_hardneg"],
        },
    )

    names = [recipe.name for recipe in recipes]
    assert names == [
        "incumbent_teacher_bag",
        "incumbent_quality_rank",
        "incumbent_anchor_mix",
        "incumbent_teacher_off_control",
    ]


def test_initial_recipes_prefer_passed_overfit_frontier(monkeypatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    research_root = artifact_dir / "research_loop"
    research_root.mkdir(parents=True)
    overfit_dir = (
        artifact_dir
        / "gates"
        / "recovery_v14_teacher_anchor_quality_focal"
        / "overfit_gate"
    )
    overfit_dir.mkdir(parents=True)
    checkpoint_path = overfit_dir / "checkpoint_best.pt"
    config = ModelConfig.rtx5000_nuscenes_teacher_bootstrap().model_copy(
        update={"freeze_image_backbone": False}
    )
    checkpoint_path.write_text("checkpoint")
    monkeypatch.setattr(
        research,
        "load_model_from_checkpoint",
        lambda *args, **kwargs: (object(), {"model_config": config.model_dump()}),
    )
    (overfit_dir / "summary.json").write_text(
        json.dumps(
            {
                "recipe": "recovery_v14_teacher_anchor_quality_focal",
                "selected_checkpoint_path": str(checkpoint_path),
                "gate_verdict": {
                    "passed": True,
                    "train_total_ratio": 0.3624,
                    "nds": 0.1553,
                    "mean_ap": 0.1992,
                    "car_ap_4m": 0.4958,
                    "nonzero_classes": 7,
                },
            }
        )
    )

    recipes = research._initial_recipes(research_root, teacher_provider_available=True)

    assert len(recipes) == 1
    assert recipes[0].name == "carryover_recovery_v14_teacher_anchor_quality_focal"
    assert recipes[0].use_teacher_provider is True
    assert recipes[0].init_checkpoint == str(checkpoint_path)
    assert recipes[0].loss_mode == "quality_focal"
    assert recipes[0].enable_teacher_distillation is False


def test_initial_recipes_prefer_previous_mini_incumbent_over_overfit_frontier(
    monkeypatch, tmp_path: Path
) -> None:
    artifact_dir = tmp_path / "artifacts"
    research_root = artifact_dir / "research_loop"
    research_root.mkdir(parents=True)
    summary_path = research_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "selected_record": {
                    "recipe": "mini_teacher_quality",
                    "config": ModelConfig.rtx5000_nuscenes_teacher_bootstrap().model_dump(),
                    "batch_size": 2,
                    "grad_accum_steps": 1,
                    "lr": 1e-4,
                    "epochs": 6,
                    "num_workers": 4,
                    "score_threshold": 0.05,
                    "top_k": 64,
                    "use_teacher_provider": True,
                    "loss_mode": "quality_focal",
                }
            }
        )
    )
    overfit_dir = (
        artifact_dir
        / "gates"
        / "recovery_v14_teacher_anchor_quality_focal"
        / "overfit_gate"
    )
    overfit_dir.mkdir(parents=True)
    checkpoint_path = overfit_dir / "checkpoint_best.pt"
    config = ModelConfig.rtx5000_nuscenes_teacher_bootstrap().model_copy(
        update={"freeze_image_backbone": False}
    )
    checkpoint_path.write_text("checkpoint")
    monkeypatch.setattr(
        research,
        "load_model_from_checkpoint",
        lambda *args, **kwargs: (object(), {"model_config": config.model_dump()}),
    )
    (overfit_dir / "summary.json").write_text(
        json.dumps(
            {
                "recipe": "recovery_v14_teacher_anchor_quality_focal",
                "selected_checkpoint_path": str(checkpoint_path),
                "gate_verdict": {
                    "passed": True,
                    "train_total_ratio": 0.3624,
                    "nds": 0.1553,
                    "mean_ap": 0.1992,
                    "car_ap_4m": 0.4958,
                    "nonzero_classes": 7,
                },
            }
        )
    )

    recipes = research._initial_recipes(research_root, teacher_provider_available=True)

    assert recipes[0].name == "carryover_mini_teacher_quality"
    assert recipes[0].parent_recipe == "mini_teacher_quality"


def test_initial_recipes_find_passed_overfit_frontier_in_canonical_repo_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    artifact_dir = repo_root / "artifacts"
    research_root = artifact_dir / "research_v15" / "research_loop"
    research_root.mkdir(parents=True)
    overfit_dir = (
        artifact_dir
        / "gates"
        / "recovery_v14_teacher_anchor_quality_focal"
        / "overfit_gate"
    )
    overfit_dir.mkdir(parents=True)
    checkpoint_path = overfit_dir / "checkpoint_best.pt"
    checkpoint_path.write_text("checkpoint")
    config = ModelConfig.rtx5000_nuscenes_teacher_bootstrap().model_copy(
        update={"freeze_image_backbone": False}
    )
    monkeypatch.setattr(research, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        research,
        "load_model_from_checkpoint",
        lambda *args, **kwargs: (object(), {"model_config": config.model_dump()}),
    )
    (overfit_dir / "summary.json").write_text(
        json.dumps(
            {
                "recipe": "recovery_v14_teacher_anchor_quality_focal",
                "selected_checkpoint_path": str(checkpoint_path),
                "gate_verdict": {
                    "passed": True,
                    "train_total_ratio": 0.3624,
                    "nds": 0.1553,
                    "mean_ap": 0.1992,
                    "car_ap_4m": 0.4958,
                    "nonzero_classes": 7,
                },
            }
        )
    )

    recipes = research._initial_recipes(research_root, teacher_provider_available=True)

    assert len(recipes) == 1
    assert recipes[0].name == "carryover_recovery_v14_teacher_anchor_quality_focal"
    assert recipes[0].init_checkpoint == str(checkpoint_path)


def test_warm_start_checkpoint_for_recipe_only_applies_to_compatible_exploit_recipe() -> None:
    incumbent_recipe = research.ResearchRecipe(
        name="incumbent",
        note="incumbent",
        hypothesis="incumbent",
        mutation_reason="incumbent",
        config=ModelConfig.rtx5000_nuscenes_baseline(),
        stage="explore",
    )
    exploit_recipe = research.ResearchRecipe(
        name="incumbent_query_boost",
        note="exploit",
        hypothesis="exploit",
        mutation_reason="exploit",
        config=ModelConfig.rtx5000_nuscenes_baseline().model_copy(update={"q_2d": 112}),
        stage="exploit",
        parent_recipe="incumbent",
    )
    incompatible_recipe = research.ResearchRecipe(
        name="incumbent_effb0",
        note="exploit",
        hypothesis="exploit",
        mutation_reason="exploit",
        config=ModelConfig.rtx5000_nuscenes_baseline().model_copy(
            update={"image_backbone": "efficientnet_b0"}
        ),
        stage="exploit",
        parent_recipe="incumbent",
    )
    teacher_recipe = research.ResearchRecipe(
        name="incumbent_teacher_kd",
        note="exploit",
        hypothesis="exploit",
        mutation_reason="exploit",
        config=ModelConfig.rtx5000_nuscenes_baseline(),
        stage="exploit",
        parent_recipe="incumbent",
        use_teacher_provider=True,
    )
    incumbent_record = {"checkpoint_path": "/tmp/incumbent.pt"}

    assert (
        research._warm_start_checkpoint_for_recipe(
            exploit_recipe,
            incumbent_recipe,
            incumbent_record,
        )
        == "/tmp/incumbent.pt"
    )
    assert (
        research._warm_start_checkpoint_for_recipe(
            incompatible_recipe,
            incumbent_recipe,
            incumbent_record,
        )
        is None
    )
    assert (
        research._warm_start_checkpoint_for_recipe(
            teacher_recipe,
            incumbent_recipe,
            incumbent_record,
        )
        == "/tmp/incumbent.pt"
    )


def test_teacher_lift_compares_teacher_kd_and_teacher_seed() -> None:
    records = [
        {
            "run_id": 1,
            "recipe": "baseline",
            "status": "completed",
            "use_teacher_provider": False,
            "teacher_seed_mode": "off",
            "evaluation": _fake_eval(0.01, 0.001),
            "val": {"total": 20.0},
        },
        {
            "run_id": 2,
            "recipe": "baseline_teacher_kd",
            "status": "completed",
            "use_teacher_provider": True,
            "teacher_seed_mode": "off",
            "evaluation": _fake_eval(0.025, 0.002),
            "val": {"total": 19.0},
        },
        {
            "run_id": 3,
            "recipe": "baseline_teacher_seed",
            "status": "completed",
            "use_teacher_provider": True,
            "teacher_seed_mode": "replace_lidar",
            "evaluation": _fake_eval(0.04, 0.003),
            "val": {"total": 18.5},
        },
    ]

    teacher_lift = research._teacher_lift(records)

    assert teacher_lift["paired"] is True
    assert teacher_lift["passed"] is True
    assert teacher_lift["baseline_recipe"] == "baseline"
    assert teacher_lift["teacher_recipe"] == "baseline_teacher_seed"
    assert teacher_lift["comparisons"]["teacher_kd"]["nds"] == 0.025
    assert teacher_lift["comparisons"]["teacher_seed"]["nds"] == 0.04
    assert teacher_lift["comparisons"]["teacher_ref_seed"]["nds"] == 0.04


def test_exploitation_candidates_prioritize_teacher_paths_when_teacher_available() -> None:
    incumbent = research.ResearchRecipe(
        name="carryover_recipe",
        note="incumbent",
        hypothesis="incumbent",
        mutation_reason="incumbent",
        config=ModelConfig.rtx5000_nuscenes_baseline(),
        stage="baseline",
    )
    incumbent_record = {
        "recipe": incumbent.name,
        "source_mix": {"lidar": 0.33, "proposal": 0.50, "global": 0.17},
        "checkpoint_path": "/tmp/incumbent.pt",
        "evaluation": _fake_eval(0.02, 0.001),
        "val": {"total": 20.0},
    }
    teacher_provider = research.TeacherProviderConfig(
        kind="cache",
        cache_dir="/tmp/cache",
    )

    candidates = research._build_exploitation_recipes(
        incumbent,
        incumbent_record,
        teacher_provider,
        remaining_budget=3,
    )
    assert [candidate.name for candidate in candidates] == [
        f"{incumbent.name}_teacher_seed",
        f"{incumbent.name}_teacher_kd",
        f"{incumbent.name}_quality_focal",
    ]
    assert candidates[0].enable_teacher_distillation is False
    assert candidates[0].config.router_mode == "anchor_first"
    assert candidates[0].top_k_candidates == (16, 32, 64)
    assert candidates[1].enable_teacher_distillation is True


def test_exploitation_candidates_do_not_repeat_active_loss_mode() -> None:
    incumbent = research.ResearchRecipe(
        name="carryover_recipe",
        note="incumbent",
        hypothesis="incumbent",
        mutation_reason="incumbent",
        config=ModelConfig.rtx5000_nuscenes_teacher_bootstrap(),
        stage="baseline",
        use_teacher_provider=True,
        loss_mode="quality_focal",
        enable_teacher_distillation=False,
    )
    incumbent_record = {
        "recipe": incumbent.name,
        "source_mix": {"lidar": 0.48, "proposal": 0.36, "global": 0.16},
        "checkpoint_path": "/tmp/incumbent.pt",
        "evaluation": _fake_eval(0.08, 0.04, car_ap_4m=0.05),
        "val": {"total": 18.0},
    }
    teacher_provider = research.TeacherProviderConfig(kind="cache", cache_dir="/tmp/cache")

    candidates = research._build_exploitation_recipes(
        incumbent,
        incumbent_record,
        teacher_provider,
        remaining_budget=9,
    )

    candidate_names = [candidate.name for candidate in candidates]
    assert f"{incumbent.name}_quality_focal" not in candidate_names
    assert f"{incumbent.name}_query_boost" in candidate_names
    assert f"{incumbent.name}_lr_down" in candidate_names
    assert f"{incumbent.name}_focal_hardneg" in candidate_names


def test_exploitation_candidates_add_winner_line_finegrid_and_teacher_quality_plus() -> None:
    incumbent = research.ResearchRecipe(
        name="carryover_recipe",
        note="incumbent",
        hypothesis="incumbent",
        mutation_reason="incumbent",
        config=ModelConfig.rtx5000_nuscenes_teacher_bootstrap().model_copy(
            update={"ranking_mode": "quality_class_only"}
        ),
        stage="baseline",
        use_teacher_provider=True,
        loss_mode="quality_focal",
        enable_teacher_distillation=False,
        teacher_anchor_quality_class_weight=0.35,
        teacher_region_objectness_weight=0.10,
        teacher_region_class_weight=0.10,
    )
    incumbent_record = {
        "recipe": incumbent.name,
        "source_mix": {"lidar": 0.8571, "proposal": 0.0, "global": 0.1429},
        "checkpoint_path": "/tmp/incumbent.pt",
        "evaluation": _fake_eval(0.1746, 0.1717, car_ap_4m=0.4303),
        "val": {"total": 11.0},
    }
    teacher_provider = research.TeacherProviderConfig(kind="cache", cache_dir="/tmp/cache")

    candidates = research._build_exploitation_recipes(
        incumbent,
        incumbent_record,
        teacher_provider,
        remaining_budget=8,
    )

    by_name = {candidate.name: candidate for candidate in candidates}
    finegrid = by_name[f"{incumbent.name}_quality_rank_finegrid"]
    assert finegrid.skip_training is True
    assert finegrid.top_k_candidates == (40, 48, 56, 64)
    teacher_plus = by_name[f"{incumbent.name}_teacher_quality_plus"]
    assert teacher_plus.teacher_anchor_quality_class_weight >= 0.45
    assert teacher_plus.teacher_region_objectness_weight >= 0.12


def test_exploitation_candidates_add_teacher_region_and_augmentation_for_teacher_runs() -> None:
    incumbent = research.ResearchRecipe(
        name="carryover_recipe",
        note="incumbent",
        hypothesis="incumbent",
        mutation_reason="incumbent",
        config=ModelConfig.rtx5000_nuscenes_teacher_bootstrap(),
        stage="baseline",
        use_teacher_provider=True,
        loss_mode="quality_focal",
        enable_teacher_distillation=False,
    )
    incumbent_record = {
        "recipe": incumbent.name,
        "source_mix": {"lidar": 1.0, "proposal": 0.0, "global": 0.0},
        "checkpoint_path": "/tmp/incumbent.pt",
        "evaluation": _fake_eval(0.12, 0.09, car_ap_4m=0.35),
        "val": {"total": 12.0},
    }
    teacher_provider = research.TeacherProviderConfig(kind="cache", cache_dir="/tmp/cache")

    candidates = research._build_exploitation_recipes(
        incumbent,
        incumbent_record,
        teacher_provider,
        remaining_budget=8,
    )

    candidate_names = [candidate.name for candidate in candidates]
    assert f"{incumbent.name}_teacher_region" in candidate_names
    assert f"{incumbent.name}_teacher_region_aug" in candidate_names


def test_query_boost_recipe_preserves_active_loss_mode() -> None:
    incumbent = research.ResearchRecipe(
        name="carryover_recipe",
        note="incumbent",
        hypothesis="incumbent",
        mutation_reason="incumbent",
        config=ModelConfig.rtx5000_nuscenes_teacher_bootstrap(),
        stage="baseline",
        use_teacher_provider=True,
        loss_mode="quality_focal",
        enable_teacher_distillation=False,
    )

    boosted = research._make_query_boost_recipe(
        incumbent,
        source_mix={"lidar": 1.0, "proposal": 0.0, "global": 0.0},
    )

    assert boosted.loss_mode == "quality_focal"


def test_exploitation_candidates_add_teacher_off_control_and_anchor_mix_for_lidar_collapse(
) -> None:
    incumbent = research.ResearchRecipe(
        name="carryover_recipe",
        note="incumbent",
        hypothesis="incumbent",
        mutation_reason="incumbent",
        config=ModelConfig.rtx5000_nuscenes_teacher_bootstrap(),
        stage="baseline",
        use_teacher_provider=True,
        loss_mode="quality_focal",
        enable_teacher_distillation=False,
    )
    incumbent_record = {
        "recipe": incumbent.name,
        "source_mix": {"lidar": 1.0, "proposal": 0.0, "global": 0.0},
        "checkpoint_path": "/tmp/incumbent.pt",
        "evaluation": _fake_eval(0.14, 0.18, car_ap_4m=0.60),
        "val": {"total": 10.0},
    }
    teacher_provider = research.TeacherProviderConfig(kind="cache", cache_dir="/tmp/cache")

    candidates = research._build_exploitation_recipes(
        incumbent,
        incumbent_record,
        teacher_provider,
        remaining_budget=10,
    )

    candidate_names = [candidate.name for candidate in candidates]
    assert f"{incumbent.name}_teacher_off_control" in candidate_names
    assert f"{incumbent.name}_anchor_mix" in candidate_names


def test_frontier_exploitation_candidates_stay_on_frontier_family() -> None:
    incumbent = research._frontier_vits16_recipe(teacher_provider_available=True)
    incumbent_record = {
        "recipe": incumbent.name,
        "source_mix": {"lidar": 0.55, "proposal": 0.30, "global": 0.15},
        "checkpoint_path": "/tmp/frontier.pt",
        "evaluation": _fake_eval(0.12, 0.10, car_ap_4m=0.25),
        "val": {"total": 11.4},
    }
    teacher_provider = research.TeacherProviderConfig(kind="cache", cache_dir="/tmp/cache")

    candidates = research._build_exploitation_recipes(
        incumbent,
        incumbent_record,
        teacher_provider,
        remaining_budget=5,
        boss_policy={
            "force_priority_only": True,
            "priority_tags": [
                "quality_rank_finegrid",
                "world_aligned_distillation",
                "sam21_offline_support",
                "dino_v3",
            ],
            "suppress_tags": ["query_boost", "lr_down", "teacher_bag", "anchor_mix"],
        },
    )

    candidate_names = [candidate.name for candidate in candidates]
    assert f"{incumbent.name}_official_guardrail" in candidate_names
    assert f"{incumbent.name}_world_distill" in candidate_names
    assert f"{incumbent.name}_no_sam2" in candidate_names
    assert f"{incumbent.name}_vitb16" in candidate_names
    assert all("query_boost" not in name for name in candidate_names)
    assert all("teacher_bag" not in name for name in candidate_names)


def test_boss_policy_suppresses_losing_branches_for_quality_rank_winner(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "research_vx" / "research_loop"
    artifact_root.mkdir(parents=True)

    latest = _fake_selected_record(
        recipe="quality_rank_incumbent",
        nds=0.1746,
        mean_ap=0.1717,
        val_total=11.0563,
        boxes_mean=32.0,
        lidar_share=0.8571,
        proposal_share=0.0,
    )
    latest["use_teacher_provider"] = True
    latest["teacher_anchor_quality_class_weight"] = 0.35
    latest["config"] = (
        ModelConfig.rtx5000_nuscenes_teacher_bootstrap()
        .model_copy(
            update={
                "teacher_seed_mode": "replace_lidar",
                "ranking_mode": "quality_class_only",
            }
        )
        .model_dump()
    )

    policy = research._boss_policy_from_history(artifact_root, extra_record=latest)

    assert policy["force_priority_only"] is True
    assert policy["priority_tags"] == [
        "quality_rank_finegrid",
        "teacher_quality_plus",
        "teacher_off_control",
    ]
    assert "teacher_bag" in policy["suppress_tags"]
    assert "anchor_mix" in policy["suppress_tags"]


def test_scale_gate_verdict_blocks_pathological_prediction_geometry() -> None:
    promoted_record = {
        "evaluation": _fake_eval(0.02, 0.001, car_ap_4m=0.01),
        "source_mix": {"lidar": 0.33, "proposal": 0.50, "global": 0.17},
        "source_mix_diagnostics": {
            "average": {"lidar": 0.33, "proposal": 0.50, "global": 0.17},
            "per_batch": [{"lidar": 0.33, "proposal": 0.50, "global": 0.17}] * 8,
            "batches_measured": 8,
        },
        "benchmark": {"mean_ms": 20.0},
        "prediction_geometry": {
            "boxes_per_sample_mean": 111.0,
            "boxes_per_sample_p95": 112.0,
            "boxes_per_sample_max": 112.0,
            "ego_translation_norm_p99": 1833.0,
            "ego_translation_norm_max": 2042.0,
        },
    }
    records = [
        {
            "status": "completed",
            "use_teacher_provider": False,
            "teacher_seed_mode": "off",
            "evaluation": _fake_eval(0.01, 0.001),
            "val": {"total": 20.0},
        },
        promoted_record,
    ]

    verdict = research._scale_gate_verdict(promoted_record, records)

    assert verdict["authorized"] is False
    assert verdict["gates"]["geometry_sanity"]["passed"] is False
    assert verdict["gates"]["geometry_sanity"]["boxes_per_sample_mean"] == 111.0
    assert verdict["gates"]["geometry_sanity"]["ego_translation_norm_p99"] == 1833.0
