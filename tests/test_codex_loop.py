from __future__ import annotations

from pathlib import Path

from tsqbev.codex_loop import run_codex_loop


def test_codex_loop_runs_one_cycle(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {
        "search": 0,
        "promote": 0,
        "memory": 0,
        "brief": 0,
        "supervisor": 0,
    }

    def fake_search(**kwargs):
        calls["search"] += 1
        candidate_root = tmp_path / "harness_v2" / "candidates" / "candidate_001"
        candidate_root.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_root / "candidate.py"
        candidate_path.write_text("CANDIDATE_METADATA={}\n", encoding="utf-8")
        return {
            "search_root": str(tmp_path / "harness_v2" / "search" / "run_001"),
            "best_candidate_path": str(candidate_path),
        }

    def fake_promote(**kwargs):
        calls["promote"] += 1
        return {"summary": {"status": "promoted"}}

    def fake_sync_memory(*, artifact_dir):
        calls["memory"] += 1
        return {"ok": True, "artifact_dir": str(artifact_dir)}

    def fake_safe_sync(repo_root):
        calls["memory"] += 1
        return {"ok": True, "repo_root": str(repo_root)}

    def fake_brief(repo_root, *, persist_log):
        calls["brief"] += 1
        return {"ok": True, "persist_log": persist_log}

    def fake_supervisor(**kwargs):
        calls["supervisor"] += 1
        return {"status": "completed", "artifact_dir": str(kwargs["artifact_dir"])}

    monkeypatch.setattr("tsqbev.codex_loop.run_harness_search", fake_search)
    monkeypatch.setattr("tsqbev.codex_loop.run_harness_promote", fake_promote)
    monkeypatch.setattr("tsqbev.codex_loop.sync_harness_memory", fake_sync_memory)
    monkeypatch.setattr("tsqbev.codex_loop.safe_sync_research_memory", fake_safe_sync)
    monkeypatch.setattr("tsqbev.codex_loop.safe_build_research_brief", fake_brief)
    monkeypatch.setattr("tsqbev.codex_loop.run_research_supervisor", fake_supervisor)
    monkeypatch.setattr("tsqbev.codex_loop.time.sleep", lambda *_args, **_kwargs: None)

    result = run_codex_loop(
        dataroot=tmp_path / "nuscenes",
        artifact_dir=tmp_path / "artifacts" / "codex_loop",
        harness_root=tmp_path / "artifacts" / "harness_v2",
        max_cycles=1,
        sleep_seconds=0,
    )

    assert result["status"] == "completed"
    assert calls["search"] == 1
    assert calls["promote"] == 1
    assert calls["memory"] == 1
    assert calls["supervisor"] == 1
    assert (tmp_path / "artifacts" / "codex_loop" / "state.json").exists()
