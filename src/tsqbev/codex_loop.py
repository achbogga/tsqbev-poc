"""Persistent Codex-style research loop with executable repo law."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tsqbev.harness_v2 import (
    DEFAULT_HARNESS_ROOT,
    DEFAULT_PROPOSAL_PATH,
    render_harness_report,
    run_harness_promote,
    run_harness_search,
)
from tsqbev.research_memory import REPO_ROOT
from tsqbev.research_supervisor import run_research_supervisor
from tsqbev.teacher_backends import TeacherProviderConfig

DEFAULT_CODEX_LOOP_ROOT = REPO_ROOT / "artifacts" / "codex_loop"
DEFAULT_LOOP_REPORT = REPO_ROOT / "docs" / "reports" / "codex_loop.md"
DEFAULT_HEARTBEAT_PATHNAME = "heartbeat.json"
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0
DEFAULT_MEMORY_SYNC_SERVICE = "tsqbev-memory-sync.service"
DEFAULT_MEMORY_SYNC_FALLBACK_SCRIPT = (
    REPO_ROOT / "research" / "scripts" / "run_memory_sync_service.sh"
)
_TRIVIAL_INFRA_MARKERS = (
    "docker",
    "conda",
    "pip",
    "onnxruntime",
    "wheel",
    "module not found",
    "filenotfounderror",
    "no such file",
    "permission denied",
    "timeout",
    "temporary failure",
    "connection reset",
)


def _timestamp_tag() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _log(message: str) -> None:
    timestamp = datetime.now(tz=UTC).isoformat()
    print(f"[codex-loop] {timestamp} {message}", flush=True)


def _classify_exception(exc: Exception) -> str:
    text = repr(exc).lower()
    if any(marker in text for marker in _TRIVIAL_INFRA_MARKERS):
        return "trivial_infra"
    return "execution_error"


def _render_loop_report(state: dict[str, Any], *, report_path: Path) -> Path:
    lines = [
        "# Codex Loop",
        f"_Generated: `{state['generated_at_utc']}`_",
        "",
        f"- status: `{state['status']}`",
        f"- artifact_root: `{state['artifact_root']}`",
        f"- harness_root: `{state['harness_root']}`",
        f"- cycles_started: `{state['cycles_started']}`",
        f"- cycles_completed: `{state['cycles_completed']}`",
        f"- last_cycle_root: `{state.get('last_cycle_root') or '-'}`",
        f"- active_phase: `{state.get('active_phase') or '-'}`",
        f"- last_error_kind: `{state.get('last_error_kind') or '-'}`",
        "",
        "## Notes",
    ]
    notes = state.get("notes", [])
    if isinstance(notes, list) and notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- none")
    return _write_text(report_path, "\n".join(lines).rstrip() + "\n")


def _persist_state(
    state_path: Path,
    report_path: Path,
    *,
    status: str,
    artifact_root: Path,
    harness_root: Path,
    dataroot: Path,
    cycles_started: int,
    cycles_completed: int,
    last_cycle_root: str | None,
    active_phase: str | None,
    active_phase_started_at_utc: str | None,
    last_error_kind: str | None,
    notes: list[str],
) -> dict[str, Any]:
    state = {
        "status": status,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "artifact_root": str(artifact_root),
        "harness_root": str(harness_root),
        "dataset_root": str(dataroot),
        "cycles_started": cycles_started,
        "cycles_completed": cycles_completed,
        "last_cycle_root": last_cycle_root,
        "active_phase": active_phase,
        "active_phase_started_at_utc": active_phase_started_at_utc,
        "last_error_kind": last_error_kind,
        "notes": notes,
    }
    _write_json(state_path, state)
    _render_loop_report(state, report_path=report_path)
    return state


def _persist_cycle_phase(cycle_root: Path, *, phase: str, notes: list[str]) -> None:
    _write_json(
        cycle_root / "phase.json",
        {
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            "phase": phase,
            "notes": notes,
        },
    )


def _write_heartbeat(
    heartbeat_path: Path,
    *,
    artifact_root: Path,
    harness_root: Path,
    dataroot: Path,
    cycles_started: int,
    cycles_completed: int,
    last_cycle_root: str | None,
    active_phase: str | None,
    active_phase_started_at_utc: str | None,
    last_error_kind: str | None,
    notes: list[str],
) -> None:
    _write_json(
        heartbeat_path,
        {
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            "pid": os.getpid(),
            "artifact_root": str(artifact_root),
            "harness_root": str(harness_root),
            "dataset_root": str(dataroot),
            "cycles_started": cycles_started,
            "cycles_completed": cycles_completed,
            "last_cycle_root": last_cycle_root,
            "active_phase": active_phase,
            "active_phase_started_at_utc": active_phase_started_at_utc,
            "last_error_kind": last_error_kind,
            "notes": notes,
        },
    )


def _service_load_state(unit_name: str) -> str:
    completed = subprocess.run(
        ["systemctl", "--user", "show", unit_name, "--property=LoadState", "--value"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "not-found"
    return completed.stdout.strip() or "unknown"


def _request_background_memory_sync(
    *,
    artifact_root: Path,
    harness_root: Path,
) -> dict[str, Any]:
    request_log = artifact_root / "memory_sync.request.log"
    unit_state = _service_load_state(DEFAULT_MEMORY_SYNC_SERVICE)
    if unit_state not in {"not-found", "masked"}:
        completed = subprocess.run(
            ["systemctl", "--user", "start", DEFAULT_MEMORY_SYNC_SERVICE],
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "mode": "systemd",
            "unit": DEFAULT_MEMORY_SYNC_SERVICE,
            "requested": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

    if DEFAULT_MEMORY_SYNC_FALLBACK_SCRIPT.exists():
        request_log.parent.mkdir(parents=True, exist_ok=True)
        with request_log.open("a", encoding="utf-8") as stream:
            subprocess.Popen(
                [
                    str(DEFAULT_MEMORY_SYNC_FALLBACK_SCRIPT),
                    str(artifact_root),
                    str(harness_root),
                ],
                cwd=str(REPO_ROOT),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return {
            "mode": "fallback_script",
            "script": str(DEFAULT_MEMORY_SYNC_FALLBACK_SCRIPT),
            "requested": True,
        }

    fallback_log = artifact_root / "memory_sync_fallback.log"
    fallback_log.parent.mkdir(parents=True, exist_ok=True)
    with fallback_log.open("a", encoding="utf-8") as stream:
        subprocess.Popen(
            [
                str(REPO_ROOT / ".venv" / "bin" / "tsqbev"),
                "harness-memory-sync",
                "--artifact-dir",
                str(harness_root),
            ],
            cwd=str(REPO_ROOT),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {
        "mode": "fallback_subprocess",
        "requested": True,
        "command": "tsqbev harness-memory-sync",
    }


def run_codex_loop(
    *,
    dataroot: Path,
    artifact_dir: str | Path = DEFAULT_CODEX_LOOP_ROOT,
    harness_root: str | Path = DEFAULT_HARNESS_ROOT,
    proposal_path: str | Path | None = DEFAULT_PROPOSAL_PATH,
    device: str | None = None,
    max_experiments: int = 5,
    teacher_provider_config: TeacherProviderConfig | None = None,
    max_cycles: int | None = None,
    search_iterations: int = 1,
    sleep_seconds: int = 30,
    wait_poll_seconds: int = 20,
    git_publish: bool = False,
    git_remote: str = "origin",
    git_branch: str | None = None,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    harness_root_path = Path(harness_root)
    proposal = None if proposal_path is None else Path(proposal_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    os.environ["TSQBEV_HARNESS_ROOT"] = str(harness_root_path)
    os.environ["TSQBEV_SUPERVISOR_USE_PROMOTED_HARNESS"] = "1"

    state_path = artifact_root / "state.json"
    report_path = DEFAULT_LOOP_REPORT
    heartbeat_path = artifact_root / DEFAULT_HEARTBEAT_PATHNAME
    cycles_started = 0
    cycles_completed = 0
    last_cycle_root: str | None = None
    last_error_kind: str | None = None
    notes: list[str] = []
    active_phase_started_at_utc = datetime.now(tz=UTC).isoformat()
    runtime_snapshot: dict[str, Any] = {
        "artifact_root": artifact_root,
        "harness_root": harness_root_path,
        "dataroot": dataroot,
        "cycles_started": cycles_started,
        "cycles_completed": cycles_completed,
        "last_cycle_root": last_cycle_root,
        "active_phase": None,
        "active_phase_started_at_utc": active_phase_started_at_utc,
        "last_error_kind": last_error_kind,
        "notes": notes,
    }
    heartbeat_stop = threading.Event()

    def _update_runtime_snapshot(
        *,
        active_phase: str | None,
        notes_value: list[str],
    ) -> None:
        previous_phase = runtime_snapshot.get("active_phase")
        if active_phase != previous_phase:
            runtime_snapshot["active_phase_started_at_utc"] = datetime.now(
                tz=UTC
            ).isoformat()
        runtime_snapshot.update(
            {
                "cycles_started": cycles_started,
                "cycles_completed": cycles_completed,
                "last_cycle_root": last_cycle_root,
                "active_phase": active_phase,
                "last_error_kind": last_error_kind,
                "notes": list(notes_value),
            }
        )

    def _heartbeat_loop() -> None:
        while not heartbeat_stop.wait(DEFAULT_HEARTBEAT_INTERVAL_S):
            _write_heartbeat(
                heartbeat_path,
                artifact_root=artifact_root,
                harness_root=harness_root_path,
                dataroot=dataroot,
                cycles_started=int(runtime_snapshot["cycles_started"]),
                cycles_completed=int(runtime_snapshot["cycles_completed"]),
                last_cycle_root=runtime_snapshot["last_cycle_root"],
                active_phase=runtime_snapshot["active_phase"],
                active_phase_started_at_utc=runtime_snapshot[
                    "active_phase_started_at_utc"
                ],
                last_error_kind=runtime_snapshot["last_error_kind"],
                notes=list(runtime_snapshot["notes"]),
            )

    _update_runtime_snapshot(active_phase="starting", notes_value=notes)
    _write_heartbeat(
        heartbeat_path,
        artifact_root=artifact_root,
        harness_root=harness_root_path,
        dataroot=dataroot,
        cycles_started=cycles_started,
        cycles_completed=cycles_completed,
        last_cycle_root=last_cycle_root,
        active_phase="starting",
        active_phase_started_at_utc=runtime_snapshot["active_phase_started_at_utc"],
        last_error_kind=last_error_kind,
        notes=notes,
    )
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        name="codex-loop-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        while max_cycles is None or cycles_started < max_cycles:
            cycles_started += 1
            cycle_root = artifact_root / f"cycle_{cycles_started:03d}_{_timestamp_tag()}"
            cycle_root.mkdir(parents=True, exist_ok=True)
            last_cycle_root = str(cycle_root)
            cycle_notes: list[str] = []
            _log(f"cycle={cycles_started} started")
            _update_runtime_snapshot(active_phase="starting_cycle", notes_value=notes)
            _persist_state(
                state_path,
                report_path,
                status="running",
                artifact_root=artifact_root,
                harness_root=harness_root_path,
                dataroot=dataroot,
                cycles_started=cycles_started,
                cycles_completed=cycles_completed,
                last_cycle_root=last_cycle_root,
                active_phase="starting_cycle",
                active_phase_started_at_utc=runtime_snapshot[
                    "active_phase_started_at_utc"
                ],
                last_error_kind=last_error_kind,
                notes=notes,
            )
            _persist_cycle_phase(cycle_root, phase="starting_cycle", notes=cycle_notes)
            try:
                _log(f"cycle={cycles_started} phase=harness_search")
                _update_runtime_snapshot(active_phase="harness_search", notes_value=notes)
                _persist_state(
                    state_path,
                    report_path,
                    status="running",
                    artifact_root=artifact_root,
                    harness_root=harness_root_path,
                    dataroot=dataroot,
                    cycles_started=cycles_started,
                    cycles_completed=cycles_completed,
                    last_cycle_root=last_cycle_root,
                    active_phase="harness_search",
                    active_phase_started_at_utc=runtime_snapshot[
                        "active_phase_started_at_utc"
                    ],
                    last_error_kind=last_error_kind,
                    notes=notes,
                )
                _persist_cycle_phase(cycle_root, phase="harness_search", notes=cycle_notes)
                search = run_harness_search(
                    artifact_dir=harness_root_path,
                    proposal_path=proposal,
                    iterations=max(search_iterations, 1),
                )
                best_candidate_path = Path(str(search["best_candidate_path"]))
                cycle_notes.append(f"best_candidate={best_candidate_path.parent.name}")
                _log(
                    f"cycle={cycles_started} phase=harness_promote best_candidate="
                    f"{best_candidate_path.parent.name}"
                )
                _update_runtime_snapshot(active_phase="harness_promote", notes_value=notes)
                _persist_state(
                    state_path,
                    report_path,
                    status="running",
                    artifact_root=artifact_root,
                    harness_root=harness_root_path,
                    dataroot=dataroot,
                    cycles_started=cycles_started,
                    cycles_completed=cycles_completed,
                    last_cycle_root=last_cycle_root,
                    active_phase="harness_promote",
                    active_phase_started_at_utc=runtime_snapshot[
                        "active_phase_started_at_utc"
                    ],
                    last_error_kind=last_error_kind,
                    notes=notes,
                )
                _persist_cycle_phase(cycle_root, phase="harness_promote", notes=cycle_notes)
                promotion = run_harness_promote(
                    artifact_dir=harness_root_path,
                    candidate_path=best_candidate_path,
                    proposal_path=proposal,
                )
                cycle_notes.append(f"promotion_status={promotion['summary']['status']}")
                _log(
                    f"cycle={cycles_started} phase=memory_refresh_request promotion_status="
                    f"{promotion['summary']['status']}"
                )
                _update_runtime_snapshot(
                    active_phase="memory_refresh_request",
                    notes_value=notes,
                )
                _persist_state(
                    state_path,
                    report_path,
                    status="running",
                    artifact_root=artifact_root,
                    harness_root=harness_root_path,
                    dataroot=dataroot,
                    cycles_started=cycles_started,
                    cycles_completed=cycles_completed,
                    last_cycle_root=last_cycle_root,
                    active_phase="memory_refresh_request",
                    active_phase_started_at_utc=runtime_snapshot[
                        "active_phase_started_at_utc"
                    ],
                    last_error_kind=last_error_kind,
                    notes=notes,
                )
                _persist_cycle_phase(
                    cycle_root,
                    phase="memory_refresh_request",
                    notes=cycle_notes,
                )
                render_harness_report(artifact_dir=harness_root_path)
                memory_request = _request_background_memory_sync(
                    artifact_root=artifact_root,
                    harness_root=harness_root_path,
                )
                cycle_notes.append(f"memory_request_mode={memory_request.get('mode', 'unknown')}")
                cycle_notes.append(f"memory_request_ok={memory_request.get('requested', False)}")
                _log(
                    f"cycle={cycles_started} phase=memory_refresh_queued "
                    f"mode={memory_request.get('mode', 'unknown')} "
                    f"requested={memory_request.get('requested', False)}"
                )
                _log(f"cycle={cycles_started} phase=supervisor")
                _update_runtime_snapshot(active_phase="supervisor", notes_value=notes)
                _persist_state(
                    state_path,
                    report_path,
                    status="running",
                    artifact_root=artifact_root,
                    harness_root=harness_root_path,
                    dataroot=dataroot,
                    cycles_started=cycles_started,
                    cycles_completed=cycles_completed,
                    last_cycle_root=last_cycle_root,
                    active_phase="supervisor",
                    active_phase_started_at_utc=runtime_snapshot[
                        "active_phase_started_at_utc"
                    ],
                    last_error_kind=last_error_kind,
                    notes=notes,
                )
                _persist_cycle_phase(cycle_root, phase="supervisor", notes=cycle_notes)
                supervisor_summary = run_research_supervisor(
                    dataroot=dataroot,
                    artifact_dir=artifact_root / "supervisor",
                    device=device,
                    max_experiments=max_experiments,
                    teacher_provider_config=teacher_provider_config,
                    max_invocations=1,
                    sleep_seconds=0,
                    wait_poll_seconds=wait_poll_seconds,
                    git_publish=git_publish,
                    git_remote=git_remote,
                    git_branch=git_branch,
                    proposal_path=proposal,
                )
                cycles_completed += 1
                last_error_kind = None
                cycle_summary = {
                    "status": "completed",
                    "generated_at_utc": datetime.now(tz=UTC).isoformat(),
                    "cycle": cycles_started,
                    "notes": cycle_notes,
                    "search_root": search["search_root"],
                    "best_candidate_path": str(best_candidate_path),
                    "promotion_status": promotion["summary"]["status"],
                    "supervisor_summary": supervisor_summary,
                }
                _write_json(cycle_root / "summary.json", cycle_summary)
                notes = cycle_notes[-6:]
                _update_runtime_snapshot(active_phase="sleep", notes_value=notes)
                _log(f"cycle={cycles_started} completed")
            except Exception as exc:  # pragma: no cover - live loop defense
                last_error_kind = _classify_exception(exc)
                error_payload = {
                    "status": "error",
                    "generated_at_utc": datetime.now(tz=UTC).isoformat(),
                    "cycle": cycles_started,
                    "error_kind": last_error_kind,
                    "error": repr(exc),
                    "notes": cycle_notes,
                }
                _write_json(cycle_root / "error.json", error_payload)
                notes = [*cycle_notes, f"error_kind={last_error_kind}"][-6:]
                _update_runtime_snapshot(active_phase="error", notes_value=notes)
                _log(f"cycle={cycles_started} error_kind={last_error_kind} error={exc!r}")

            state = _persist_state(
                state_path,
                report_path,
                status=(
                    "sleeping"
                    if max_cycles is None or cycles_started < max_cycles
                    else "completed"
                ),
                artifact_root=artifact_root,
                harness_root=harness_root_path,
                dataroot=dataroot,
                cycles_started=cycles_started,
                cycles_completed=cycles_completed,
                last_cycle_root=last_cycle_root,
                active_phase="sleep",
                active_phase_started_at_utc=runtime_snapshot[
                    "active_phase_started_at_utc"
                ],
                last_error_kind=last_error_kind,
                notes=notes,
            )
            _update_runtime_snapshot(active_phase="sleep", notes_value=notes)
            if max_cycles is not None and cycles_started >= max_cycles:
                return state
            _log(f"cycle={cycles_started} sleeping_for={max(sleep_seconds, 0)}s")
            time.sleep(max(sleep_seconds, 0))
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)

    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    return final_state
