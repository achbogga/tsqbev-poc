from __future__ import annotations

import json
from pathlib import Path

from tsqbev.research_memory import (
    ResearchMemoryConfig,
    _artifact_files,
    build_research_brief,
    query_research_memory,
    sync_research_memory,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_repo_fixture(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    _write(repo_root / "README.md", "# tsqbev-poc\n\nCurrent reset target is dense BEV fusion.\n")
    _write(repo_root / "program.md", "Status: enabled.\n")
    _write(repo_root / "AGENTS.md", "# Agent Rules\n\nUse memory-first research.\n")
    _write(
        repo_root / "docs" / "plan.md",
        "# Plan\n\nScale-up is blocked until quality improves.\n",
    )
    _write(
        repo_root / "docs" / "steering.md",
        "# Steering\n\nPrioritize dense-BEV reset evidence.\n",
    )
    _write(
        repo_root / "specs" / "004-research-loop-contract.md",
        "# Spec 004\n\nThe loop must remain append-only.\n",
    )

    results_record = {
        "run_id": 4,
        "recipe": "mini_propheavy_mbv3_frozen_query_boost",
        "stage": "exploit",
        "parent_recipe": "mini_propheavy_mbv3_frozen",
        "status": "completed",
        "interim_decision": "advance",
        "final_decision": "promote",
        "git_sha": "deadbee",
        "hypothesis": "query boost should improve recall",
        "mutation_reason": "bounded sparse budget increase",
        "root_cause_verdict": "optimization bottleneck",
        "config": {
            "image_backbone": "mobilenet_v3_large",
            "teacher_seed_mode": "off",
            "views": 6,
        },
        "evaluation": {"nd_score": 0.0158, "mean_ap": 0.0001},
        "val": {"total": 20.1352},
        "benchmark": {"mean_ms": 17.19},
    }
    _write(
        repo_root / "artifacts" / "research_v3" / "research_loop" / "results.jsonl",
        json.dumps(results_record) + "\n",
    )

    summary = {
        "status": "completed",
        "reference_workflow": "karpathy/autoresearch",
        "selected_recipe": "mini_propheavy_mbv3_frozen_query_boost",
        "selected_record": {
            "recipe": "mini_propheavy_mbv3_frozen_query_boost",
            "config": {
                "image_backbone": "mobilenet_v3_large",
                "teacher_seed_mode": "off",
                "views": 6,
            },
            "evaluation": {"nd_score": 0.0158, "mean_ap": 0.0001},
            "val": {"total": 20.1352},
        },
        "scale_gate_verdict": {
            "authorized": False,
            "reason": "optimization gate remains blocked by weak mini generalization",
        },
        "recommended_next_steps": [
            "Continue dense-BEV reset reproduction and compare against BEVFusion.",
            "Do not scale compute until the gate is cleared.",
        ],
    }
    _write(
        repo_root / "artifacts" / "research_v3" / "research_loop" / "summary.json",
        json.dumps(summary, indent=2),
    )

    upstream = {
        "checkpoint_path": "pretrained/bevfusion-det.pth",
        "config_rel": (
            "configs/nuscenes/det/transfusion/secfpn/"
            "camera+lidar/swint_v0p075/convfuser.yaml"
        ),
        "headline_metrics": {
            "mAP": 0.6730,
            "NDS": 0.7072,
            "mATE": 0.2859,
            "mASE": 0.2559,
        },
        "status": "success",
        "upstream_repo_root": "/home/achbogga/projects/bevfusion",
    }
    _write(
        repo_root / "artifacts" / "bevfusion_repro" / "bevfusion_bbox_summary.json",
        json.dumps(upstream, indent=2),
    )
    _write(repo_root / "docs" / "reports" / "current.md", "# stale generated report\n")
    _write(
        repo_root / "artifacts" / "memory" / "brief.json",
        json.dumps({"stale": True}, indent=2),
    )
    return repo_root


def test_sync_research_memory_writes_catalog_and_manifest(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    config = ResearchMemoryConfig(
        memory_root=tmp_path / ".local" / "memory",
        artifact_root=repo_root / "artifacts" / "memory",
        reports_root=repo_root / "docs" / "reports",
        report_log_root=repo_root / "docs" / "reports" / "log",
        steering_path=repo_root / "docs" / "steering.md",
        qdrant_enabled=False,
        mem0_enabled=False,
    )

    summary = sync_research_memory(repo_root, config=config)

    assert summary["event_count"] >= 3
    assert summary["evidence_count"] > 0
    assert summary["fact_count"] >= 3
    assert (config.catalog_path).exists()
    assert (config.artifact_root / "sync_manifest.json").exists()


def test_build_research_brief_uses_indexed_state(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    config = ResearchMemoryConfig(
        memory_root=tmp_path / ".local" / "memory",
        artifact_root=repo_root / "artifacts" / "memory",
        reports_root=repo_root / "docs" / "reports",
        report_log_root=repo_root / "docs" / "reports" / "log",
        steering_path=repo_root / "docs" / "steering.md",
        qdrant_enabled=False,
        mem0_enabled=False,
    )
    sync_research_memory(repo_root, config=config)

    brief = build_research_brief(repo_root, config=config, persist_log=True)

    assert any("mini_propheavy_mbv3_frozen_query_boost" in line for line in brief.current_state)
    assert any("bevfusion:convfuser" in line for line in brief.current_state)
    assert any("Scale-up blocker" in line for line in brief.open_blockers)
    assert (config.reports_root / "current.md").exists()
    assert any(config.report_log_root.iterdir())
    assert all("research_memory.py" not in line for line in brief.evidence_refs)


def test_query_research_memory_returns_exact_facts(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    config = ResearchMemoryConfig(
        memory_root=tmp_path / ".local" / "memory",
        artifact_root=repo_root / "artifacts" / "memory",
        reports_root=repo_root / "docs" / "reports",
        report_log_root=repo_root / "docs" / "reports" / "log",
        steering_path=repo_root / "docs" / "steering.md",
        qdrant_enabled=False,
        mem0_enabled=False,
    )
    sync_research_memory(repo_root, config=config)

    result = query_research_memory("scale-up blocked", repo_root=repo_root, config=config, limit=4)

    assert result["exact_facts"]
    assert any("Scale-up is still blocked" in item["claim"] for item in result["exact_facts"])


def test_artifact_files_exclude_generated_memory_outputs(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)

    files = {path.relative_to(repo_root).as_posix() for path in _artifact_files(repo_root)}

    assert "docs/reports/current.md" not in files
    assert "artifacts/memory/brief.json" not in files
    assert "artifacts/bevfusion_repro/bevfusion_bbox_summary.json" in files
