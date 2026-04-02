"""Continuous bounded research supervisor.

References:
- Andrej Karpathy autoresearch workflow:
  https://github.com/karpathy/autoresearch
- Repo-local research loop contract:
  https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tsqbev.research import run_bounded_research_loop
from tsqbev.research_guard import ensure_research_loop_enabled
from tsqbev.research_memory import (
    REPO_ROOT,
    check_research_memory_health,
    current_git_sha,
    safe_build_research_brief,
    safe_sync_research_memory,
)
from tsqbev.teacher_backends import TeacherProviderConfig

DEFAULT_SUPERVISOR_ROOT = REPO_ROOT / "artifacts" / "autoresearch"
DEFAULT_SUPERVISOR_REPORT = REPO_ROOT / "docs" / "reports" / "autoresearch.md"


@dataclass(slots=True)
class SupervisorState:
    status: str
    generated_at_utc: str
    repo_sha: str
    current_branch: str
    dataset_root: str
    artifact_root: str
    attempted_invocations: int
    completed_invocations: int
    last_invocation_dir: str | None
    last_selected_recipe: str | None
    last_nds: float | None
    last_map: float | None
    last_publish_status: str | None
    last_publish_message: str | None
    memory_mode: str | None
    memory_embedder: str | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _timestamp_tag() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _external_research_loop_processes(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["pgrep", "-af", "tsqbev research-loop"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        return []
    current_pid = os.getpid()
    processes: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        parts = raw_line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        if pid == current_pid:
            continue
        if "research-supervisor" in cmd:
            continue
        processes.append({"pid": pid, "cmd": cmd})
    return processes


def _publish_paths_for_invocation(invocation_root: Path, repo_root: Path = REPO_ROOT) -> list[Path]:
    paths: list[Path] = []
    for relative in (
        Path("docs/reports/current.md"),
        Path("docs/reports/autoresearch.md"),
        Path("artifacts/memory/brief.json"),
        Path("artifacts/memory/sync_manifest.json"),
    ):
        candidate = repo_root / relative
        if candidate.exists():
            paths.append(relative)

    log_root = repo_root / "docs" / "reports" / "log"
    if log_root.exists():
        for log_path in sorted(log_root.glob("*.md")):
            paths.append(log_path.relative_to(repo_root))

    loop_root = invocation_root / "research_loop"
    for loop_relative in (
        "pre_run_brief.json",
        "summary.json",
        "results.tsv",
        "results.jsonl",
    ):
        candidate = loop_root / loop_relative
        if candidate.exists():
            paths.append(candidate.relative_to(repo_root))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _git_publish_generated(
    repo_root: Path,
    *,
    invocation_root: Path,
    remote: str,
    branch: str | None,
) -> dict[str, Any]:
    publish_paths = _publish_paths_for_invocation(invocation_root, repo_root=repo_root)
    if not publish_paths:
        return {
            "status": "noop",
            "message": "no publishable report artifacts found",
            "paths": [],
        }

    add_cmd = ["git", "-C", str(repo_root), "add", "--", *[str(path) for path in publish_paths]]
    add_completed = subprocess.run(add_cmd, check=False, capture_output=True, text=True)
    if add_completed.returncode != 0:
        return {
            "status": "error",
            "message": add_completed.stderr.strip() or "git add failed",
            "paths": [str(path) for path in publish_paths],
        }

    diff_cmd = ["git", "-C", str(repo_root), "diff", "--cached", "--quiet", "--"]
    diff_completed = subprocess.run(diff_cmd, check=False, capture_output=True, text=True)
    if diff_completed.returncode == 0:
        return {
            "status": "noop",
            "message": "no staged report changes to publish",
            "paths": [str(path) for path in publish_paths],
        }

    target_branch = branch or _git_current_branch(repo_root)
    commit_message = (
        "autoresearch: publish "
        f"{invocation_root.name} ({datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M UTC')})"
    )
    commit_cmd = ["git", "-C", str(repo_root), "commit", "-m", commit_message, "--"]
    commit_completed = subprocess.run(commit_cmd, check=False, capture_output=True, text=True)
    if commit_completed.returncode != 0:
        return {
            "status": "error",
            "message": commit_completed.stderr.strip() or "git commit failed",
            "paths": [str(path) for path in publish_paths],
        }

    push_cmd = ["git", "-C", str(repo_root), "push", remote, f"HEAD:{target_branch}"]
    push_completed = subprocess.run(push_cmd, check=False, capture_output=True, text=True)
    if push_completed.returncode != 0:
        return {
            "status": "push_failed",
            "message": push_completed.stderr.strip() or "git push failed",
            "paths": [str(path) for path in publish_paths],
        }

    return {
        "status": "published",
        "message": commit_message,
        "paths": [str(path) for path in publish_paths],
        "branch": target_branch,
        "remote": remote,
    }


def _render_supervisor_report(
    state: SupervisorState,
    *,
    latest_brief_path: Path,
    ledger_path: Path,
    stop_path: Path,
) -> str:
    brief_rel = latest_brief_path.relative_to(REPO_ROOT)
    ledger_rel = ledger_path.relative_to(REPO_ROOT)
    stop_rel = stop_path.relative_to(REPO_ROOT)
    lines = [
        "# Autoresearch Supervisor",
        f"_Generated: `{state.generated_at_utc}`_",
        "",
        "## Status",
        f"- status: `{state.status}`",
        f"- branch: `{state.current_branch}`",
        f"- repo sha: `{state.repo_sha}`",
        f"- dataset root: `{state.dataset_root}`",
        f"- artifact root: `{state.artifact_root}`",
        f"- attempted invocations: `{state.attempted_invocations}`",
        f"- completed invocations: `{state.completed_invocations}`",
        f"- memory mode: `{state.memory_mode or 'unknown'}`",
        f"- memory embedder: `{state.memory_embedder or 'unknown'}`",
        f"- last invocation dir: `{state.last_invocation_dir or '-'}`",
        f"- last selected recipe: `{state.last_selected_recipe or '-'}`",
        f"- last NDS: `{state.last_nds if state.last_nds is not None else '-'}`",
        f"- last mAP: `{state.last_map if state.last_map is not None else '-'}`",
        f"- last publish status: `{state.last_publish_status or '-'}`",
        f"- last publish message: `{state.last_publish_message or '-'}`",
        "",
        "## Notes",
        *[f"- {note}" for note in state.notes],
        "",
        "## Pointers",
        f"- current PI brief: [{brief_rel}]({brief_rel})",
        f"- supervisor ledger: [{ledger_rel}]({ledger_rel})",
        f"- supervisor stop file: `{stop_rel}`",
    ]
    return "\n".join(lines) + "\n"


def _write_supervisor_outputs(
    state: SupervisorState,
    *,
    artifact_root: Path,
    report_path: Path,
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = artifact_root / "state.json"
    state_path.write_text(json.dumps(state.to_dict(), indent=2))
    report_text = _render_supervisor_report(
        state,
        latest_brief_path=REPO_ROOT / "docs" / "reports" / "current.md",
        ledger_path=artifact_root / "ledger.jsonl",
        stop_path=artifact_root / "STOP",
    )
    report_path.write_text(report_text)


def run_research_supervisor(
    *,
    dataroot: str | Path,
    artifact_dir: str | Path = DEFAULT_SUPERVISOR_ROOT,
    device: str | None = None,
    max_experiments: int = 5,
    teacher_provider_config: TeacherProviderConfig | None = None,
    max_invocations: int | None = None,
    sleep_seconds: int = 30,
    wait_poll_seconds: int = 20,
    git_publish: bool = True,
    git_remote: str = "origin",
    git_branch: str | None = None,
) -> dict[str, Any]:
    """Run a continuous bounded research supervisor with memory + report publishing."""

    ensure_research_loop_enabled()
    dataset_root = Path(dataroot)
    supervisor_root = Path(artifact_dir)
    supervisor_root.mkdir(parents=True, exist_ok=True)
    ledger_path = supervisor_root / "ledger.jsonl"
    stop_path = supervisor_root / "STOP"

    attempted_invocations = 0
    completed_invocations = 0
    last_invocation_dir: Path | None = None
    last_selected_recipe: str | None = None
    last_nds: float | None = None
    last_map: float | None = None
    last_publish_status: str | None = None
    last_publish_message: str | None = None

    while max_invocations is None or attempted_invocations < max_invocations:
        memory_health = check_research_memory_health(REPO_ROOT)
        memory_mode = None
        memory_embedder = None
        if isinstance(memory_health, dict):
            qdrant = memory_health.get("qdrant")
            if isinstance(qdrant, dict):
                memory_mode = str(qdrant.get("mode"))
                memory_embedder = str(qdrant.get("embedder_provider"))

        if stop_path.exists():
            state = SupervisorState(
                status="stopped",
                generated_at_utc=datetime.now(tz=UTC).isoformat(),
                repo_sha=current_git_sha(REPO_ROOT),
                current_branch=_git_current_branch(REPO_ROOT),
                dataset_root=str(dataset_root),
                artifact_root=str(supervisor_root),
                attempted_invocations=attempted_invocations,
                completed_invocations=completed_invocations,
                last_invocation_dir=str(last_invocation_dir) if last_invocation_dir else None,
                last_selected_recipe=last_selected_recipe,
                last_nds=last_nds,
                last_map=last_map,
                last_publish_status=last_publish_status,
                last_publish_message=last_publish_message,
                memory_mode=memory_mode,
                memory_embedder=memory_embedder,
                notes=["Stop file present; supervisor exiting cleanly."],
            )
            _write_supervisor_outputs(
                state,
                artifact_root=supervisor_root,
                report_path=DEFAULT_SUPERVISOR_REPORT,
            )
            return state.to_dict()

        external_runs = _external_research_loop_processes(REPO_ROOT)
        if external_runs:
            state = SupervisorState(
                status="waiting_external_run",
                generated_at_utc=datetime.now(tz=UTC).isoformat(),
                repo_sha=current_git_sha(REPO_ROOT),
                current_branch=_git_current_branch(REPO_ROOT),
                dataset_root=str(dataset_root),
                artifact_root=str(supervisor_root),
                attempted_invocations=attempted_invocations,
                completed_invocations=completed_invocations,
                last_invocation_dir=str(last_invocation_dir) if last_invocation_dir else None,
                last_selected_recipe=last_selected_recipe,
                last_nds=last_nds,
                last_map=last_map,
                last_publish_status=last_publish_status,
                last_publish_message=last_publish_message,
                memory_mode=memory_mode,
                memory_embedder=memory_embedder,
                notes=[
                    "External `tsqbev research-loop` process detected; waiting instead of "
                    "contending for the same GPU.",
                    *[
                        f"pid `{item['pid']}`: `{item['cmd']}`"
                        for item in external_runs[:3]
                    ],
                ],
            )
            _write_supervisor_outputs(
                state,
                artifact_root=supervisor_root,
                report_path=DEFAULT_SUPERVISOR_REPORT,
            )
            time.sleep(wait_poll_seconds)
            continue

        attempted_invocations += 1
        invocation_root = (
            supervisor_root / f"invocation_{attempted_invocations:03d}_{_timestamp_tag()}"
        )
        last_invocation_dir = invocation_root
        safe_sync_research_memory(REPO_ROOT)
        safe_build_research_brief(REPO_ROOT, persist_log=False)

        started_at = datetime.now(tz=UTC).isoformat()
        invocation_status = "completed"
        notes: list[str] = [f"started invocation `{invocation_root.name}` at `{started_at}`"]
        try:
            summary = run_bounded_research_loop(
                dataroot=dataset_root,
                artifact_dir=invocation_root,
                device=device,
                max_experiments=max_experiments,
                teacher_provider_config=teacher_provider_config,
            )
            selected_record = summary.get("selected_record", {})
            if isinstance(selected_record, dict):
                evaluation = selected_record.get("evaluation", {})
                if isinstance(evaluation, dict):
                    last_nds = float(evaluation.get("nd_score", 0.0))
                    last_map = float(evaluation.get("mean_ap", 0.0))
                last_selected_recipe = str(selected_record.get("recipe"))
            completed_invocations += 1
        except Exception as exc:
            summary = {
                "status": "crash",
                "error": repr(exc),
                "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            }
            invocation_status = "crash"
            notes.append(f"invocation crashed: `{exc!r}`")

        safe_sync_research_memory(REPO_ROOT)
        safe_build_research_brief(REPO_ROOT, persist_log=True)

        if git_publish:
            publish_result = _git_publish_generated(
                REPO_ROOT,
                invocation_root=invocation_root,
                remote=git_remote,
                branch=git_branch,
            )
        else:
            publish_result = {"status": "disabled", "message": "git publishing disabled"}
        last_publish_status = str(publish_result.get("status"))
        last_publish_message = str(publish_result.get("message"))

        entry = {
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            "repo_sha": current_git_sha(REPO_ROOT),
            "branch": _git_current_branch(REPO_ROOT),
            "status": invocation_status,
            "invocation": attempted_invocations,
            "invocation_root": str(invocation_root),
            "selected_recipe": last_selected_recipe,
            "nds": last_nds,
            "mean_ap": last_map,
            "publish": publish_result,
            "summary": summary,
        }
        _append_jsonl(ledger_path, entry)
        notes.append(
            f"finished invocation `{invocation_root.name}` with publish status "
            f"`{last_publish_status}`"
        )

        status = (
            "running"
            if max_invocations is None or attempted_invocations < max_invocations
            else "completed"
        )
        state = SupervisorState(
            status=status,
            generated_at_utc=datetime.now(tz=UTC).isoformat(),
            repo_sha=current_git_sha(REPO_ROOT),
            current_branch=_git_current_branch(REPO_ROOT),
            dataset_root=str(dataset_root),
            artifact_root=str(supervisor_root),
            attempted_invocations=attempted_invocations,
            completed_invocations=completed_invocations,
            last_invocation_dir=str(last_invocation_dir) if last_invocation_dir else None,
            last_selected_recipe=last_selected_recipe,
            last_nds=last_nds,
            last_map=last_map,
            last_publish_status=last_publish_status,
            last_publish_message=last_publish_message,
            memory_mode=memory_mode,
            memory_embedder=memory_embedder,
            notes=notes,
        )
        _write_supervisor_outputs(
            state,
            artifact_root=supervisor_root,
            report_path=DEFAULT_SUPERVISOR_REPORT,
        )

        if max_invocations is not None and attempted_invocations >= max_invocations:
            return state.to_dict()
        time.sleep(sleep_seconds)

    return {
        "status": "completed",
        "attempted_invocations": attempted_invocations,
        "completed_invocations": completed_invocations,
        "artifact_root": str(supervisor_root),
    }
