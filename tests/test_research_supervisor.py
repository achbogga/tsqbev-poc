from __future__ import annotations

import json
from pathlib import Path

from tsqbev import research_supervisor


def test_publish_paths_for_invocation_include_reports_and_loop_outputs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    invocation_root = repo_root / "artifacts" / "autoresearch" / "invocation_001"
    (repo_root / "docs" / "reports" / "log").mkdir(parents=True)
    (repo_root / "artifacts" / "memory").mkdir(parents=True)
    (invocation_root / "research_loop").mkdir(parents=True)

    for path in (
        repo_root / "docs" / "reports" / "current.md",
        repo_root / "docs" / "reports" / "autoresearch.md",
        repo_root / "docs" / "reports" / "log" / "20260402-000000.md",
        repo_root / "artifacts" / "memory" / "brief.json",
        repo_root / "artifacts" / "memory" / "sync_manifest.json",
        invocation_root / "research_loop" / "pre_run_brief.json",
        invocation_root / "research_loop" / "results.jsonl",
        invocation_root / "research_loop" / "results.tsv",
        invocation_root / "research_loop" / "summary.json",
    ):
        path.write_text("x")

    publish_paths = research_supervisor._publish_paths_for_invocation(
        invocation_root,
        repo_root=repo_root,
    )

    assert Path("docs/reports/current.md") in publish_paths
    assert Path("docs/reports/autoresearch.md") in publish_paths
    assert Path("artifacts/memory/brief.json") in publish_paths
    assert Path("artifacts/memory/sync_manifest.json") in publish_paths
    assert Path("docs/reports/log/20260402-000000.md") in publish_paths
    assert Path("artifacts/autoresearch/invocation_001/research_loop/summary.json") in publish_paths


def test_run_research_supervisor_waits_then_runs_and_writes_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs" / "reports").mkdir(parents=True)
    (repo_root / "artifacts" / "memory").mkdir(parents=True)
    (repo_root / "docs" / "reports" / "current.md").write_text("# current\n")
    (repo_root / "artifacts" / "memory" / "brief.json").write_text("{}")
    (repo_root / "artifacts" / "memory" / "sync_manifest.json").write_text("{}")
    supervisor_root = repo_root / "artifacts" / "autoresearch"
    dataset_root = tmp_path / "nuscenes"
    dataset_root.mkdir()

    monkeypatch.setattr(research_supervisor, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        research_supervisor,
        "DEFAULT_SUPERVISOR_REPORT",
        repo_root / "docs" / "reports" / "autoresearch.md",
    )
    monkeypatch.setattr(research_supervisor, "ensure_research_loop_enabled", lambda: None)
    monkeypatch.setattr(
        research_supervisor,
        "check_research_memory_health",
        lambda *args, **kwargs: {
            "qdrant": {"mode": "server", "embedder_provider": "hash"},
        },
    )
    monkeypatch.setattr(research_supervisor, "current_git_sha", lambda *args, **kwargs: "abc1234")
    monkeypatch.setattr(research_supervisor, "_git_current_branch", lambda *args, **kwargs: "main")
    monkeypatch.setattr(
        research_supervisor,
        "safe_sync_research_memory",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        research_supervisor,
        "safe_build_research_brief",
        lambda *args, **kwargs: {"status": "ok"},
    )

    external_calls = {"count": 0}

    def fake_external(*args, **kwargs):
        external_calls["count"] += 1
        if external_calls["count"] == 1:
            return [{"pid": 999, "cmd": "uv run tsqbev research-loop"}]
        return []

    monkeypatch.setattr(research_supervisor, "_external_research_loop_processes", fake_external)
    monkeypatch.setattr(research_supervisor.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        research_supervisor,
        "run_bounded_research_loop",
        lambda **kwargs: {
            "status": "completed",
            "selected_record": {
                "recipe": "carryover_recipe",
                "evaluation": {"nd_score": 0.1234, "mean_ap": 0.0567},
            },
        },
    )
    monkeypatch.setattr(
        research_supervisor,
        "_git_publish_generated",
        lambda *args, **kwargs: {"status": "published", "message": "ok"},
    )

    state = research_supervisor.run_research_supervisor(
        dataroot=dataset_root,
        artifact_dir=supervisor_root,
        max_invocations=1,
        git_publish=True,
    )

    assert state["status"] == "completed"
    assert state["attempted_invocations"] == 1
    assert state["completed_invocations"] == 1
    assert state["last_selected_recipe"] == "carryover_recipe"
    assert state["last_nds"] == 0.1234
    assert state["last_map"] == 0.0567

    report_path = repo_root / "docs" / "reports" / "autoresearch.md"
    assert report_path.exists()
    assert "Autoresearch Supervisor" in report_path.read_text()

    state_path = supervisor_root / "state.json"
    assert state_path.exists()
    persisted = json.loads(state_path.read_text())
    assert persisted["last_publish_status"] == "published"

    ledger_path = supervisor_root / "ledger.jsonl"
    assert ledger_path.exists()
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["selected_recipe"] == "carryover_recipe"
