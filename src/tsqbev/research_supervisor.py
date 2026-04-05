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

_OpenAI: Any
try:  # pragma: no cover - optional runtime dependency.
    from openai import OpenAI as _OpenAI
except ImportError:  # pragma: no cover
    _OpenAI = None
OpenAIClient: Any = _OpenAI

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
    planner_provider: str | None
    critic_provider: str | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlannerDecision:
    provider: str
    model: str | None
    active_bottleneck: str
    objective: str
    priority_tags: list[str]
    suppress_tags: list[str]
    force_priority_only: bool
    token_burn_score: int
    rationale: list[str]
    kill_conditions: list[str]

    def to_policy(self) -> dict[str, Any]:
        return {
            "priority_tags": self.priority_tags,
            "suppress_tags": self.suppress_tags,
            "force_priority_only": self.force_priority_only,
        }


@dataclass(slots=True)
class CriticDecision:
    provider: str
    model: str | None
    approved: bool
    rationale: list[str]
    supervisor_policy: dict[str, Any]


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

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key.startswith("artifacts/autoresearch/"):
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _write_first_principles_checkpoint(
    invocation_root: Path,
    brief: dict[str, Any],
) -> Path:
    invocation_root.mkdir(parents=True, exist_ok=True)
    current_state = brief.get("current_state")
    blockers = brief.get("open_blockers")
    next_steps = brief.get("recommended_next_steps")
    evidence = brief.get("evidence_refs")
    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "first_principles_questions": [
            "What is the current strongest local evidence, not the incumbent label?",
            "What bottleneck is actually active right now?",
            "What upstream or teacher evidence says the current student path is wrong "
            "or incomplete?",
            "What is the smallest bounded next move that directly targets the active bottleneck?",
            "What stopping condition would prove this branch is no longer the best ROI path?",
        ],
        "current_state": current_state if isinstance(current_state, list) else [],
        "open_blockers": blockers if isinstance(blockers, list) else [],
        "recommended_next_steps": next_steps if isinstance(next_steps, list) else [],
        "evidence_refs": evidence if isinstance(evidence, list) else [],
    }
    path = invocation_root / "first_principles_checkpoint.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("reasoner response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def _heuristic_planner(brief: dict[str, Any]) -> PlannerDecision:
    current_state = brief.get("current_state", [])
    state_text = "\n".join(str(item) for item in current_state if isinstance(item, str)).lower()
    if "joint" in state_text and "0.0000" in state_text:
        return PlannerDecision(
            provider="heuristic",
            model=None,
            active_bottleneck="joint-metric-collapse",
            objective=(
                "stop broken joint promotion and focus on official-metric-safe "
                "detection control"
            ),
            priority_tags=["quality_rank_finegrid", "teacher_quality_plus", "teacher_off_control"],
            suppress_tags=["teacher_bag", "anchor_mix", "augmentation", "lr_down", "query_boost"],
            force_priority_only=True,
            token_burn_score=-2,
            rationale=[
                "official joint detection collapsed to zero, so more multitask churn is low ROI",
                "current best local line remains teacher-quality plus "
                "quality-rank control detection",
            ],
            kill_conditions=[
                "two consecutive runs fail to beat the current detection incumbent on NDS",
                "export sanity degrades while loss improves",
            ],
        )
    return PlannerDecision(
        provider="heuristic",
        model=None,
        active_bottleneck="quality-vs-calibration-boundary",
        objective="improve official mini-val NDS without reopening dead exploit families",
        priority_tags=["quality_rank_finegrid", "teacher_quality_plus", "teacher_off_control"],
        suppress_tags=["teacher_bag", "anchor_mix", "augmentation", "lr_down"],
        force_priority_only=True,
        token_burn_score=-1,
        rationale=[
            "quality-rank and teacher-quality-plus are the only winning branches so far",
            "bag-style exploits and broad mutation fanout have underperformed",
        ],
        kill_conditions=[
            "two consecutive runs do not produce a meaningful progress verdict",
            "calibration improvements stop moving official NDS",
        ],
    )


def _openai_reasoner(prompt: str, *, model: str) -> dict[str, Any]:
    if OpenAIClient is None or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI client or key unavailable")
    client = OpenAIClient()
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    text = getattr(response, "output_text", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OpenAI response did not return text output")
    return _extract_json_object(text)


def _planner_decision_from_brief(brief: dict[str, Any]) -> PlannerDecision:
    model = os.getenv("TSQBEV_SUPERVISOR_PLANNER_MODEL", "gpt-5.4")
    prompt = (
        "You are the planner for an autonomous perception research lab. "
        "Given the current research brief, output only JSON with keys: "
        "active_bottleneck, objective, priority_tags, suppress_tags, "
        "force_priority_only, token_burn_score, rationale, kill_conditions. "
        "Use only short strings. Focus on highest-ROI next experiments.\n\n"
        f"BRIEF:\n{json.dumps(brief, indent=2, default=str)}"
    )
    try:
        payload = _openai_reasoner(prompt, model=model)
        return PlannerDecision(
            provider="openai",
            model=model,
            active_bottleneck=str(payload["active_bottleneck"]),
            objective=str(payload["objective"]),
            priority_tags=[str(item) for item in payload.get("priority_tags", [])],
            suppress_tags=[str(item) for item in payload.get("suppress_tags", [])],
            force_priority_only=bool(payload.get("force_priority_only", False)),
            token_burn_score=int(payload.get("token_burn_score", 0)),
            rationale=[str(item) for item in payload.get("rationale", [])],
            kill_conditions=[str(item) for item in payload.get("kill_conditions", [])],
        )
    except Exception:
        return _heuristic_planner(brief)


def _critic_decision_from_planner(
    brief: dict[str, Any],
    planner: PlannerDecision,
) -> CriticDecision:
    model = os.getenv("TSQBEV_SUPERVISOR_CRITIC_MODEL", "gpt-5.4-mini")
    prompt = (
        "You are the critic for an autonomous perception research lab. "
        "Review the planner decision and current brief. Output only JSON with keys: "
        "approved, rationale, priority_tags, suppress_tags, force_priority_only. "
        "Reject repeated low-ROI branches.\n\n"
        f"BRIEF:\n{json.dumps(brief, indent=2, default=str)}\n\n"
        f"PLANNER:\n{json.dumps(asdict(planner), indent=2, default=str)}"
    )
    try:
        payload = _openai_reasoner(prompt, model=model)
        priority_tags = [
            str(item) for item in payload.get("priority_tags", planner.priority_tags)
        ]
        suppress_tags = [
            str(item) for item in payload.get("suppress_tags", planner.suppress_tags)
        ]
        supervisor_policy = {
            "priority_tags": priority_tags,
            "suppress_tags": suppress_tags,
            "force_priority_only": bool(
                payload.get("force_priority_only", planner.force_priority_only)
            ),
        }
        return CriticDecision(
            provider="openai",
            model=model,
            approved=bool(payload.get("approved", True)),
            rationale=[str(item) for item in payload.get("rationale", [])],
            supervisor_policy=supervisor_policy,
        )
    except Exception:
        approved = planner.token_burn_score < 3
        rationale = list(planner.rationale)
        if not approved:
            rationale.append("token burn score exceeded the allowed threshold")
        return CriticDecision(
            provider="heuristic",
            model=None,
            approved=approved,
            rationale=rationale,
            supervisor_policy=planner.to_policy(),
        )


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
        f"- planner provider: `{state.planner_provider or 'unknown'}`",
        f"- critic provider: `{state.critic_provider or 'unknown'}`",
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
    planner_provider: str | None = None
    critic_provider: str | None = None

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
                planner_provider=planner_provider,
                critic_provider=critic_provider,
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
                planner_provider=planner_provider,
                critic_provider=critic_provider,
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
        pre_run_brief = safe_build_research_brief(REPO_ROOT, persist_log=False)
        checkpoint_path = _write_first_principles_checkpoint(invocation_root, pre_run_brief)
        planner_decision = _planner_decision_from_brief(pre_run_brief)
        critic_decision = _critic_decision_from_planner(pre_run_brief, planner_decision)
        planner_provider = planner_decision.provider
        critic_provider = critic_decision.provider
        (invocation_root / "planner_decision.json").write_text(
            json.dumps(asdict(planner_decision), indent=2)
        )
        (invocation_root / "critic_decision.json").write_text(
            json.dumps(asdict(critic_decision), indent=2)
        )

        started_at = datetime.now(tz=UTC).isoformat()
        invocation_status = "completed"
        notes: list[str] = [
            f"started invocation `{invocation_root.name}` at `{started_at}`",
            "wrote first-principles checkpoint to "
            f"`{checkpoint_path.relative_to(REPO_ROOT)}` before launch",
            f"planner provider `{planner_decision.provider}` selected bottleneck "
            f"`{planner_decision.active_bottleneck}`",
        ]
        if not critic_decision.approved:
            notes.append("critic rejected the planner proposal; skipping execution")
            invocation_status = "rejected"
            summary = {
                "status": "rejected",
                "planner_decision": asdict(planner_decision),
                "critic_decision": asdict(critic_decision),
            }
            safe_sync_research_memory(REPO_ROOT)
            safe_build_research_brief(REPO_ROOT, persist_log=True)
            publish_result = {"status": "disabled", "message": "execution skipped by critic"}
            last_publish_status = str(publish_result.get("status"))
            last_publish_message = str(publish_result.get("message"))
            entry = {
                "generated_at_utc": datetime.now(tz=UTC).isoformat(),
                "repo_sha": current_git_sha(REPO_ROOT),
                "branch": _git_current_branch(REPO_ROOT),
                "status": invocation_status,
                "invocation": attempted_invocations,
                "invocation_root": str(invocation_root),
                "first_principles_checkpoint": str(checkpoint_path),
                "planner_decision": asdict(planner_decision),
                "critic_decision": asdict(critic_decision),
                "publish": publish_result,
                "summary": summary,
            }
            _append_jsonl(ledger_path, entry)
            state = SupervisorState(
                status="running",
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
                planner_provider=planner_provider,
                critic_provider=critic_provider,
                notes=notes + critic_decision.rationale,
            )
            _write_supervisor_outputs(
                state,
                artifact_root=supervisor_root,
                report_path=DEFAULT_SUPERVISOR_REPORT,
            )
            time.sleep(sleep_seconds)
            continue
        try:
            summary = run_bounded_research_loop(
                dataroot=dataset_root,
                artifact_dir=invocation_root,
                device=device,
                max_experiments=max_experiments,
                teacher_provider_config=teacher_provider_config,
                supervisor_policy=critic_decision.supervisor_policy,
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
            "first_principles_checkpoint": str(checkpoint_path),
            "selected_recipe": last_selected_recipe,
            "nds": last_nds,
            "mean_ap": last_map,
            "publish": publish_result,
            "summary": summary,
            "planner_decision": asdict(planner_decision),
            "critic_decision": asdict(critic_decision),
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
            planner_provider=planner_provider,
            critic_provider=critic_provider,
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
