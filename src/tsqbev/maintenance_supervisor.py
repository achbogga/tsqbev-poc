"""CPU-only daily repo maintenance supervisor.

References:
- Andrej Karpathy autoresearch workflow:
  https://github.com/karpathy/autoresearch
- Repo-local memory/report contract:
  https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tsqbev.research_memory import (
    REPO_ROOT,
    current_git_sha,
    safe_build_research_brief,
    safe_sync_research_memory,
)

DEFAULT_MAINTENANCE_ROOT = REPO_ROOT / "artifacts" / "maintenance"
DEFAULT_MAINTENANCE_REPORT = REPO_ROOT / "docs" / "reports" / "maintenance.md"

KNOWN_GENERATED_PREFIXES = (
    "artifacts/memory/",
    "docs/reports/",
)


@dataclass(slots=True)
class MaintenanceState:
    status: str
    generated_at_utc: str
    repo_sha: str
    current_branch: str
    artifact_root: str
    total_runs: int
    last_run_id: str | None
    last_publish_status: str | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timestamp_tag() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _git_current_branch(repo_root: Path = REPO_ROOT) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str))
        handle.write("\n")


def _run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git_status_summary(repo_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    generated = [
        line for line in lines
        if any(line[3:].startswith(prefix) for prefix in KNOWN_GENERATED_PREFIXES if len(line) >= 4)
    ]
    non_generated = [line for line in lines if line not in generated]
    return {
        "returncode": completed.returncode,
        "dirty": bool(lines),
        "total_dirty": len(lines),
        "generated_dirty": len(generated),
        "non_generated_dirty": len(non_generated),
        "generated_paths": generated[:20],
        "non_generated_paths": non_generated[:20],
    }


def _render_maintenance_report(
    *,
    run_id: str,
    started_at_utc: str,
    finished_at_utc: str,
    git_status: dict[str, Any],
    command_results: dict[str, dict[str, Any]],
    memory_health: dict[str, Any] | None,
    report_path: Path,
    ledger_path: Path,
) -> str:
    lines = [
        "# Maintenance Supervisor",
        f"_Run: `{run_id}`_",
        "",
        "## Summary",
        f"- started: `{started_at_utc}`",
        f"- finished: `{finished_at_utc}`",
        f"- report path: `{report_path.relative_to(REPO_ROOT)}`",
        f"- ledger path: `{ledger_path.relative_to(REPO_ROOT)}`",
        "",
        "## Git Status",
        f"- dirty paths: `{git_status['total_dirty']}`",
        f"- generated-only paths: `{git_status['generated_dirty']}`",
        f"- non-generated paths: `{git_status['non_generated_dirty']}`",
    ]
    if git_status["non_generated_paths"]:
        lines.append("- non-generated dirty examples:")
        for item in git_status["non_generated_paths"]:
            lines.append(f"  - `{item}`")
    lines.extend(
        [
            "",
            "## Checks",
        ]
    )
    for name, result in command_results.items():
        status = "pass" if int(result["returncode"]) == 0 else "fail"
        lines.append(f"- `{name}`: `{status}`")
        stderr = str(result.get("stderr") or "")
        if stderr:
            lines.append(f"  stderr: `{stderr.splitlines()[-1][:180]}`")
    if memory_health is not None:
        qdrant = memory_health.get("qdrant", {})
        mem0 = memory_health.get("mem0", {})
        lines.extend(
            [
                "",
                "## Memory",
                f"- qdrant mode: `{qdrant.get('mode', 'unknown')}`",
                f"- embedder: `{qdrant.get('embedder_provider', 'unknown')}`",
                f"- mem0 enabled: `{mem0.get('enabled', 'unknown')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_supervisor_outputs(
    state: MaintenanceState,
    *,
    artifact_root: Path,
    report_path: Path,
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    (artifact_root / "state.json").write_text(json.dumps(state.to_dict(), indent=2))


def run_maintenance_once(
    *,
    artifact_dir: str | Path = DEFAULT_MAINTENANCE_ROOT,
    publish_reports: bool = False,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    ledger_path = artifact_root / "ledger.jsonl"
    run_id = _timestamp_tag()
    run_root = artifact_root / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(tz=UTC).isoformat()

    safe_sync_research_memory(REPO_ROOT)
    memory_health = None
    try:
        from tsqbev.research_memory import check_research_memory_health

        health = check_research_memory_health(REPO_ROOT)
        if isinstance(health, dict):
            memory_health = health
    except Exception:
        memory_health = None

    git_status = _git_status_summary(REPO_ROOT)
    command_results = {
        "ruff": _run_command(["uv", "run", "ruff", "check", "src", "tests"]),
        "mypy": _run_command(["uv", "run", "mypy", "src"]),
        "pytest": _run_command(["uv", "run", "pytest", "-q"]),
    }
    safe_build_research_brief(REPO_ROOT, persist_log=False)
    finished_at = datetime.now(tz=UTC).isoformat()

    report_text = _render_maintenance_report(
        run_id=run_id,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        git_status=git_status,
        command_results=command_results,
        memory_health=memory_health,
        report_path=DEFAULT_MAINTENANCE_REPORT,
        ledger_path=ledger_path,
    )
    DEFAULT_MAINTENANCE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_MAINTENANCE_REPORT.write_text(report_text)
    (run_root / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "git_status": git_status,
                "checks": command_results,
                "memory_health": memory_health,
            },
            indent=2,
        )
    )
    _append_jsonl(
        ledger_path,
        {
            "run_id": run_id,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "git_status": git_status,
            "checks": {name: result["returncode"] for name, result in command_results.items()},
        },
    )

    if publish_reports:
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "add",
                "--",
                str(DEFAULT_MAINTENANCE_REPORT.relative_to(REPO_ROOT)),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    return {
        "run_id": run_id,
        "artifact_root": str(run_root),
        "report_path": str(DEFAULT_MAINTENANCE_REPORT),
        "git_status": git_status,
        "checks": command_results,
    }


def run_maintenance_supervisor(
    *,
    artifact_dir: str | Path = DEFAULT_MAINTENANCE_ROOT,
    interval_hours: int = 24,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    stop_path = artifact_root / "STOP"
    total_runs = 0
    last_run_id: str | None = None

    while True:
        if stop_path.exists():
            state = MaintenanceState(
                status="stopped",
                generated_at_utc=datetime.now(tz=UTC).isoformat(),
                repo_sha=current_git_sha(REPO_ROOT),
                current_branch=_git_current_branch(REPO_ROOT),
                artifact_root=str(artifact_root),
                total_runs=total_runs,
                last_run_id=last_run_id,
                last_publish_status=None,
                notes=["Stop file present; maintenance supervisor exiting cleanly."],
            )
            _write_supervisor_outputs(
                state,
                artifact_root=artifact_root,
                report_path=DEFAULT_MAINTENANCE_REPORT,
            )
            return state.to_dict()

        result = run_maintenance_once(artifact_dir=artifact_root)
        total_runs += 1
        last_run_id = str(result["run_id"])
        state = MaintenanceState(
            status="sleeping",
            generated_at_utc=datetime.now(tz=UTC).isoformat(),
            repo_sha=current_git_sha(REPO_ROOT),
            current_branch=_git_current_branch(REPO_ROOT),
            artifact_root=str(artifact_root),
            total_runs=total_runs,
            last_run_id=last_run_id,
            last_publish_status=None,
            notes=[
                f"completed maintenance run `{last_run_id}`; sleeping for "
                f"`{interval_hours}` hour(s)."
            ],
        )
        _write_supervisor_outputs(
            state,
            artifact_root=artifact_root,
            report_path=DEFAULT_MAINTENANCE_REPORT,
        )
        time.sleep(max(1, interval_hours) * 3600)
