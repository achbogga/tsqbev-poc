from __future__ import annotations

import json
from pathlib import Path

from tsqbev import maintenance_supervisor as ms


def test_run_maintenance_once_writes_report_and_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ms, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        ms,
        "DEFAULT_MAINTENANCE_ROOT",
        tmp_path / "artifacts" / "maintenance",
    )
    monkeypatch.setattr(
        ms,
        "DEFAULT_MAINTENANCE_REPORT",
        tmp_path / "docs" / "reports" / "maintenance.md",
    )
    monkeypatch.setattr(ms, "safe_sync_research_memory", lambda repo_root: None)
    monkeypatch.setattr(
        ms,
        "safe_build_research_brief",
        lambda repo_root, persist_log=False: {"ok": True},
    )
    monkeypatch.setattr(ms, "current_git_sha", lambda repo_root: "deadbeef")
    monkeypatch.setattr(ms, "_git_current_branch", lambda repo_root=tmp_path: "main")
    monkeypatch.setattr(
        ms,
        "_git_status_summary",
        lambda repo_root: {
            "returncode": 0,
            "dirty": False,
            "total_dirty": 0,
            "generated_dirty": 0,
            "non_generated_dirty": 0,
            "generated_paths": [],
            "non_generated_paths": [],
        },
    )
    monkeypatch.setattr(
        ms,
        "_run_command",
        lambda command: {"command": command, "returncode": 0, "stdout": "", "stderr": ""},
    )

    result = ms.run_maintenance_once(artifact_dir=tmp_path / "artifacts" / "maintenance")

    report_path = Path(result["report_path"])
    assert report_path.exists()
    summary_path = Path(result["artifact_root"]) / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["checks"]["ruff"]["returncode"] == 0
