from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from tsqbev.codex_loop import _request_background_memory_sync, run_codex_loop


def test_codex_loop_runs_one_cycle(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {
        "search": 0,
        "promote": 0,
        "memory_request": 0,
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

    def fake_request_memory_sync(*, artifact_root, harness_root):
        calls["memory_request"] += 1
        return {
            "mode": "test",
            "requested": True,
            "artifact_root": str(artifact_root),
            "harness_root": str(harness_root),
        }

    def fake_supervisor(**kwargs):
        calls["supervisor"] += 1
        return {"status": "completed", "artifact_dir": str(kwargs["artifact_dir"])}

    monkeypatch.setattr("tsqbev.codex_loop.run_harness_search", fake_search)
    monkeypatch.setattr("tsqbev.codex_loop.run_harness_promote", fake_promote)
    monkeypatch.setattr(
        "tsqbev.codex_loop._request_background_memory_sync",
        fake_request_memory_sync,
    )
    monkeypatch.setattr(
        "tsqbev.codex_loop.render_harness_report",
        lambda *, artifact_dir: {"ok": True, "artifact_dir": str(artifact_dir)},
    )
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
    assert calls["memory_request"] == 1
    assert calls["supervisor"] == 1
    state_path = tmp_path / "artifacts" / "codex_loop" / "state.json"
    heartbeat_path = tmp_path / "artifacts" / "codex_loop" / "heartbeat.json"
    assert state_path.exists()
    assert heartbeat_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert state["active_phase_started_at_utc"]
    assert heartbeat["active_phase_started_at_utc"]


def test_request_background_memory_sync_uses_non_blocking_systemd(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_service_load_state(_unit_name: str) -> str:
        return "loaded"

    def fake_run(cmd: list[str], **kwargs) -> CompletedProcess[str]:
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "tsqbev.codex_loop._service_load_state",
        fake_service_load_state,
    )
    monkeypatch.setattr("tsqbev.codex_loop.subprocess.run", fake_run)

    result = _request_background_memory_sync(
        artifact_root=tmp_path / "artifacts",
        harness_root=tmp_path / "artifacts" / "harness_v2",
    )

    assert result["mode"] == "systemd"
    assert result["requested"] is True
    assert seen["cmd"] == [
        "systemctl",
        "--user",
        "start",
        "--no-block",
        "tsqbev-memory-sync.service",
    ]
