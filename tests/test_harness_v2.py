from __future__ import annotations

import json
from pathlib import Path

from tsqbev.harness_v2 import (
    _persist_context_summary_if_needed,
    render_harness_report,
    run_harness_benchmark,
    run_harness_promote,
    run_harness_search,
    run_harness_shadow,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_repo_fixture(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    _write(repo_root / "README.md", "# Repo\n")
    _write(repo_root / "program.md", "research loop enabled\n")
    _write(repo_root / "AGENTS.md", "# Agents\n")
    _write(repo_root / "docs" / "steering.md", "# Steering\nPrioritize frontier thesis.\n")
    _write(
        repo_root / "docs" / "paper" / "tsqbev_frontier_program.md",
        "# Frontier Program\nUse DINOv3, world latent distillation, and geometry sanity.\n",
    )
    _write(
        repo_root
        / "artifacts"
        / "research_v29_continuation_v1"
        / "research_loop"
        / "results.jsonl",
        json.dumps(
            {
                "run_id": 29,
                "recipe": "quality_rank_finegrid",
                "stage": "exploit",
                "status": "completed",
                "final_decision": "promote",
                "git_sha": "deadbee",
                "hypothesis": "finegrid thresholding",
                "mutation_reason": "incremental calibration",
                "root_cause_verdict": "incremental_progress",
                "config": {
                    "image_backbone": "mobilenet_v3_large",
                    "teacher_seed_mode": "replace_lidar",
                    "views": 6,
                },
                "evaluation": {"nd_score": 0.1833, "mean_ap": 0.1814},
                "val": {"total": 10.6925},
                "benchmark": {"mean_ms": 18.6},
            }
        )
        + "\n",
    )
    _write(
        repo_root / "artifacts" / "research_v29_continuation_v1" / "research_loop" / "summary.json",
        json.dumps(
            {
                "reference_workflow": "karpathy/autoresearch",
                "selected_recipe": "quality_rank_finegrid",
                "selected_record": {
                    "recipe": "quality_rank_finegrid",
                    "config": {
                        "image_backbone": "mobilenet_v3_large",
                        "teacher_seed_mode": "replace_lidar",
                        "views": 6,
                    },
                    "evaluation": {"nd_score": 0.1833, "mean_ap": 0.1814},
                    "val": {"total": 10.6925},
                },
                "scale_gate_verdict": {
                    "authorized": False,
                    "reason": "at least one scale gate remains unmet; do not spend 10x compute yet",
                },
                "recommended_next_steps": [
                    "run the 32-sample overfit gate before any larger-scale training",
                    "preserve the current latency envelope while chasing geometry gains",
                ],
            },
            indent=2,
        ),
    )
    _write(
        repo_root
        / "artifacts"
        / "gates"
        / "recovery_v14_teacher_anchor_quality_focal"
        / "overfit_gate"
        / "summary.json",
        json.dumps(
            {
                "subset_size": 32,
                "evaluation": {"nd_score": 0.1553, "mean_ap": 0.1992},
                "train": {"final_val_total": 13.6891},
                "gate_verdict": {
                    "passed": True,
                    "train_total_ratio": 0.3624,
                    "nds": 0.1553,
                    "mean_ap": 0.1992,
                    "car_ap_4m": 0.4958,
                    "nonzero_classes": 7,
                },
            },
            indent=2,
        ),
    )
    _write(
        repo_root / "artifacts" / "bevfusion_repro" / "bevfusion_bbox_summary.json",
        json.dumps(
            {
                "headline_metrics": {"mAP": 0.6730, "NDS": 0.7072},
                "status": "success",
            },
            indent=2,
        ),
    )
    return repo_root


def test_context_summary_persists_when_budget_is_tight(tmp_path: Path) -> None:
    task = {
        "task_id": "demo",
        "candidate_id": "candidate",
        "brief": {
            "current_state": ["x" * 200],
            "open_blockers": ["geometry"] * 3,
            "recommended_next_steps": ["step"] * 3,
            "evidence_refs": ["ref"] * 3,
        },
    }
    summary = _persist_context_summary_if_needed(
        artifact_root=tmp_path,
        phase="benchmark",
        task=task,
        budget_chars=100,
    )
    assert summary is not None
    assert (tmp_path / "benchmark_context_summary.json").exists()


def test_harness_benchmark_scores_candidate(monkeypatch, tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    monkeypatch.setattr("tsqbev.harness_v2.REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "tsqbev.harness_v2.DEFAULT_PROPOSAL_PATH",
        repo_root / "docs" / "paper" / "tsqbev_frontier_program.md",
    )
    result = run_harness_benchmark(artifact_dir=repo_root / "artifacts" / "harness_v2")
    assert result["scorecard"]["total_score"] >= 0.0
    assert result["scorecard"]["total_score"] <= 100.0
    assert Path(result["benchmark_root"]).exists()


def test_harness_search_builds_leaderboard(monkeypatch, tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    monkeypatch.setattr("tsqbev.harness_v2.REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "tsqbev.harness_v2.DEFAULT_PROPOSAL_PATH",
        repo_root / "docs" / "paper" / "tsqbev_frontier_program.md",
    )
    result = run_harness_search(
        artifact_dir=repo_root / "artifacts" / "harness_v2",
        iterations=2,
        provider="heuristic",
    )
    assert len(result["leaderboard"]) >= 2
    assert Path(result["search_root"]).exists()


def test_harness_shadow_and_promote_flow(monkeypatch, tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    monkeypatch.setattr("tsqbev.harness_v2.REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "tsqbev.harness_v2.DEFAULT_PROPOSAL_PATH",
        repo_root / "docs" / "paper" / "tsqbev_frontier_program.md",
    )
    search = run_harness_search(
        artifact_dir=repo_root / "artifacts" / "harness_v2",
        iterations=2,
        provider="heuristic",
    )
    best_candidate = Path(search["best_candidate_path"])
    shadow = run_harness_shadow(
        artifact_dir=repo_root / "artifacts" / "harness_v2",
        candidate_path=best_candidate,
    )
    assert Path(shadow["shadow_root"]).exists()
    promotion = run_harness_promote(
        artifact_dir=repo_root / "artifacts" / "harness_v2",
        candidate_path=best_candidate,
    )
    assert Path(promotion["promotion_root"]).exists()


def test_render_harness_report(monkeypatch, tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    monkeypatch.setattr("tsqbev.harness_v2.REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "tsqbev.harness_v2.DEFAULT_PROPOSAL_PATH",
        repo_root / "docs" / "paper" / "tsqbev_frontier_program.md",
    )
    run_harness_search(
        artifact_dir=repo_root / "artifacts" / "harness_v2",
        iterations=1,
        provider="heuristic",
    )
    report = render_harness_report(
        artifact_dir=repo_root / "artifacts" / "harness_v2",
        report_path=repo_root / "docs" / "reports" / "harness_v2.md",
    )
    assert Path(report["report_path"]).exists()
