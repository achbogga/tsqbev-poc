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
        "mini_propheavy_effb0_frozen_query_boost": 20.0,
        "mini_propheavy_effb0_frozen_lr_down": 22.5,
    }
    evals = {
        "mini_balanced_mbv3_frozen": _fake_eval(0.0, 0.0),
        "mini_propheavy_mbv3_frozen": _fake_eval(0.0, 0.0),
        "mini_propheavy_effb0_frozen": _fake_eval(0.01, 0.0),
        "mini_propheavy_effb0_frozen_query_boost": _fake_eval(0.03, 0.01, car_ap_4m=0.02),
        "mini_propheavy_effb0_frozen_lr_down": _fake_eval(0.015, 0.0),
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

    def fake_export(**kwargs: object) -> Path:
        output_path = Path(str(kwargs["output_path"]))
        output_path.write_text("{}")
        return output_path

    monkeypatch.setattr(research, "export_nuscenes_predictions", fake_export)

    def fake_eval_predictions(**kwargs: object) -> dict[str, object]:
        result_path = Path(str(kwargs["result_path"]))
        return evals[result_path.parent.name]

    monkeypatch.setattr(research, "evaluate_nuscenes_predictions", fake_eval_predictions)
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
        max_experiments=5,
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
    assert len(records) == 5
    promoted = [record for record in records if record["final_decision"] == "promote"]
    assert len(promoted) == 1
    assert promoted[0]["recipe"] == "mini_propheavy_effb0_frozen_query_boost"
    assert promoted[0]["final_rank"] == 1

    tsv_lines = tsv_path.read_text().splitlines()
    assert tsv_lines[0].startswith("run_id\trecipe\tstage")
    assert len(tsv_lines) == 6
    assert jsonl_snapshots[:5] == [1, 2, 3, 4, 5]
    assert tsv_snapshots[:5] == [1, 2, 3, 4, 5]
    assert jsonl_snapshots[-1] == 5
    assert tsv_snapshots[-1] == 5

    manifest_path = (
        artifact_dir
        / "research_loop"
        / "mini_propheavy_effb0_frozen_query_boost"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["git_sha"] == "deadbee"
    assert manifest["recipe"]["recipe"] == "mini_propheavy_effb0_frozen_query_boost"


def test_initial_recipes_can_carry_forward_previous_incumbent(tmp_path: Path) -> None:
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


def test_initial_recipes_insert_teacher_kd_when_teacher_is_available(tmp_path: Path) -> None:
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
    assert recipes[1].name == "carryover_mini_propheavy_effb0_frozen_teacher_kd"
    assert recipes[1].use_teacher_provider is True


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
        f"{incumbent.name}_teacher_kd",
        f"{incumbent.name}_teacher_seed",
        f"{incumbent.name}_query_boost",
    ]


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
            "translation_norm_p99": 1833.0,
            "translation_norm_max": 2042.0,
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
