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

from tsqbev.harness_v2 import DEFAULT_HARNESS_ROOT, load_promoted_harness_plan
from tsqbev.research import ResearchProposal, run_bounded_research_loop
from tsqbev.research_guard import ensure_research_loop_enabled
from tsqbev.research_memory import (
    REPO_ROOT,
    ResearchMemoryConfig,
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
DEFAULT_PROPOSAL_PATH = REPO_ROOT / "docs" / "paper" / "tsqbev_frontier_program.md"
DEFAULT_SYNC_TIMEOUT_SECONDS = 180
DEFAULT_BRIEF_TIMEOUT_SECONDS = 120


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
    active_phase: str | None = None
    active_checklist_item: str | None = None
    planner_bottleneck: str | None = None
    planner_objective: str | None = None
    planner_decision_path: str | None = None
    critic_decision_path: str | None = None
    proposal_path: str | None = None

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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _timestamp_tag() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _linux_process_name(pid: int) -> str | None:
    status_path = Path(f"/proc/{pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Name:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


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
        if _linux_process_name(pid) == "pt_data_worker":
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


def _tsqbev_cli() -> list[str]:
    candidate = REPO_ROOT / ".venv" / "bin" / "tsqbev"
    if candidate.exists():
        return [str(candidate)]
    return ["uv", "run", "tsqbev"]


def _run_maintenance_cli(command: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [*_tsqbev_cli(), *command],
            check=False,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "command": command,
            "timeout_seconds": timeout_seconds,
            "duration_s": time.monotonic() - started,
            "stdout": (exc.stdout or "")[-2000:],
            "stderr": (exc.stderr or "")[-2000:],
        }
    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "command": command,
        "returncode": completed.returncode,
        "duration_s": time.monotonic() - started,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def _load_proposal_context(proposal_path: Path | None, *, max_chars: int = 7000) -> str:
    if proposal_path is None or not proposal_path.exists():
        return ""
    text = proposal_path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    compact = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "\n\n[truncated]"


def _extract_selected_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    selected = summary.get("selected_record", {})
    if not isinstance(selected, dict):
        return {}
    evaluation = selected.get("evaluation", {})
    val_payload = selected.get("val", {})
    benchmark = selected.get("benchmark", {})
    if not isinstance(evaluation, dict):
        evaluation = {}
    if not isinstance(val_payload, dict):
        val_payload = {}
    if not isinstance(benchmark, dict):
        benchmark = {}
    return {
        "selected_recipe": selected.get("recipe"),
        "nds": evaluation.get("nd_score"),
        "mean_ap": evaluation.get("mean_ap"),
        "val_total": val_payload.get("total"),
        "latency_ms": benchmark.get("mean_ms"),
    }


def _write_context_refresh_summary(
    invocation_root: Path,
    *,
    phase: str,
    proposal_path: Path | None,
    proposal_context: str,
    brief: dict[str, Any],
    planner_decision: PlannerDecision,
    critic_decision: CriticDecision,
    notes: list[str],
    summary: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    invocation_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "supervisor_invocation",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "invocation_id": invocation_root.name,
        "phase": phase,
        "proposal_path": (
            str(proposal_path.relative_to(REPO_ROOT))
            if proposal_path is not None and proposal_path.exists()
            else None
        ),
        "proposal_context_excerpt": proposal_context,
        "brief": {
            "current_state": brief.get("current_state", []),
            "open_blockers": brief.get("open_blockers", []),
            "recommended_next_steps": brief.get("recommended_next_steps", []),
            "evidence_refs": brief.get("evidence_refs", []),
        },
        "planner_decision": asdict(planner_decision),
        "critic_decision": asdict(critic_decision),
        "notes": notes,
        "loop_summary": summary or {},
        "selected_metrics": _extract_selected_metrics(summary or {}),
    }
    brief_payload = payload["brief"]
    assert isinstance(brief_payload, dict)
    brief_current_state = list(brief_payload.get("current_state", []))
    brief_open_blockers = list(brief_payload.get("open_blockers", []))
    brief_next_steps = list(brief_payload.get("recommended_next_steps", []))
    selected_metrics = payload["selected_metrics"]
    assert isinstance(selected_metrics, dict)
    json_path = invocation_root / f"{phase}_context_summary.json"
    md_path = invocation_root / f"{phase}_context_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_lines = [
        f"# {phase.replace('_', ' ').title()} Context Summary",
        f"_Generated: `{payload['generated_at_utc']}`_",
        "",
        f"- proposal path: `{payload['proposal_path'] or '-'}`",
        f"- planner bottleneck: `{planner_decision.active_bottleneck}`",
        f"- planner objective: `{planner_decision.objective}`",
        f"- critic approved: `{critic_decision.approved}`",
        "",
        "## Proposal Context",
        proposal_context or "No proposal context loaded.",
        "",
        "## Brief Current State",
        *[f"- {line}" for line in brief_current_state],
        "",
        "## Open Blockers",
        *[f"- {line}" for line in brief_open_blockers],
        "",
        "## Recommended Next Steps",
        *[f"- {line}" for line in brief_next_steps],
        "",
        "## Planner Rationale",
        *[f"- {line}" for line in planner_decision.rationale],
        "",
        "## Critic Rationale",
        *[f"- {line}" for line in critic_decision.rationale],
        "",
        "## Notes",
        *[f"- {line}" for line in notes],
    ]
    if selected_metrics:
        md_lines.extend(
            [
                "",
                "## Selected Metrics",
                *[
                    f"- {key}: `{value}`"
                    for key, value in selected_metrics.items()
                    if value is not None
                ],
            ]
        )
    md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    return json_path, md_path


def _build_supervisor_proposal(
    planner_decision: PlannerDecision,
    critic_decision: CriticDecision,
    *,
    invocation_root: Path,
) -> ResearchProposal:
    policy = critic_decision.supervisor_policy
    launch_tags = [
        str(tag) for tag in policy.get("priority_tags", planner_decision.priority_tags)
    ]
    exploitation_tags = launch_tags.copy()
    if (
        "quality_rank_finegrid" in exploitation_tags
        and "teacher_quality_plus" not in exploitation_tags
    ):
        exploitation_tags.append("teacher_quality_plus")
    if (
        planner_decision.active_bottleneck == "joint-metric-collapse"
        and "teacher_off_control" not in exploitation_tags
    ):
        exploitation_tags.append("teacher_off_control")
    suppress_tags = [
        str(tag) for tag in policy.get("suppress_tags", planner_decision.suppress_tags)
    ]
    force_tags_only = bool(
        policy.get("force_priority_only", planner_decision.force_priority_only)
    )
    return ResearchProposal(
        proposal_id=invocation_root.name,
        objective=planner_decision.objective,
        targeted_bottleneck=planner_decision.active_bottleneck,
        launch_tags=launch_tags,
        exploitation_tags=exploitation_tags,
        suppress_tags=suppress_tags,
        rationale=[*planner_decision.rationale, *critic_decision.rationale],
        kill_conditions=planner_decision.kill_conditions,
        force_tags_only=force_tags_only,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("reasoner response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


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


def _planner_decision_from_brief(
    brief: dict[str, Any],
    *,
    proposal_context: str = "",
    proposal_path: Path | None = None,
    runtime_root: Path | None = None,
) -> PlannerDecision:
    if _env_bool("TSQBEV_SUPERVISOR_USE_PROMOTED_HARNESS", True):
        try:
            harness_payload = load_promoted_harness_plan(
                brief=brief,
                proposal_path=proposal_path or DEFAULT_PROPOSAL_PATH,
                artifact_dir=Path(os.getenv("TSQBEV_HARNESS_ROOT", str(DEFAULT_HARNESS_ROOT))),
                budget_chars=int(
                    os.getenv("TSQBEV_HARNESS_CONTEXT_BUDGET_CHARS", "16000")
                ),
                runtime_root=runtime_root,
            )
        except Exception:
            harness_payload = None
        if harness_payload is not None:
            plan = harness_payload.get("plan", {})
            if isinstance(plan, dict):
                return PlannerDecision(
                    provider=f"harness:{harness_payload['candidate_id']}",
                    model=None,
                    active_bottleneck=str(
                        plan.get("targeted_bottleneck", "frontier-hard-pivot")
                    ),
                    objective=str(
                        plan.get(
                            "objective",
                            "execute the promoted frontier harness on the hard-pivot stack",
                        )
                    ),
                    priority_tags=_as_str_list(plan.get("priority_tags", [])),
                    suppress_tags=_as_str_list(plan.get("suppress_tags", [])),
                    force_priority_only=bool(plan.get("force_priority_only", True)),
                    token_burn_score=0,
                    rationale=_as_str_list(plan.get("rationale", [])),
                    kill_conditions=_as_str_list(plan.get("kill_conditions", [])),
                )
    model = os.getenv("TSQBEV_SUPERVISOR_PLANNER_MODEL", "gpt-5.4")
    prompt = (
        "You are the planner for an autonomous perception research lab. "
        "Given the current research brief, output only JSON with keys: "
        "active_bottleneck, objective, priority_tags, suppress_tags, "
        "force_priority_only, token_burn_score, rationale, kill_conditions. "
        "Use only short strings. Focus on highest-ROI next experiments. Treat the proposal "
        "context as binding unless the brief contains direct counter-evidence.\n\n"
        f"PROPOSAL_CONTEXT:\n{proposal_context}\n\n"
        f"BRIEF:\n{json.dumps(brief, indent=2, default=str)}"
    )
    try:
        payload = _openai_reasoner(prompt, model=model)
        return PlannerDecision(
            provider="openai",
            model=model,
            active_bottleneck=str(payload["active_bottleneck"]),
            objective=str(payload["objective"]),
            priority_tags=_as_str_list(payload.get("priority_tags", [])),
            suppress_tags=_as_str_list(payload.get("suppress_tags", [])),
            force_priority_only=bool(payload.get("force_priority_only", False)),
            token_burn_score=int(payload.get("token_burn_score", 0)),
            rationale=_as_str_list(payload.get("rationale", [])),
            kill_conditions=_as_str_list(payload.get("kill_conditions", [])),
        )
    except Exception:
        return _heuristic_planner(brief)


def _critic_decision_from_planner(
    brief: dict[str, Any],
    planner: PlannerDecision,
    *,
    proposal_context: str = "",
) -> CriticDecision:
    model = os.getenv("TSQBEV_SUPERVISOR_CRITIC_MODEL", "gpt-5.4-mini")
    prompt = (
        "You are the critic for an autonomous perception research lab. "
        "Review the planner decision and current brief. Output only JSON with keys: "
        "approved, rationale, priority_tags, suppress_tags, force_priority_only. "
        "Reject repeated low-ROI branches. Treat the proposal context as the active thesis and "
        "reject plans that drift away from it without strong evidence.\n\n"
        f"PROPOSAL_CONTEXT:\n{proposal_context}\n\n"
        f"BRIEF:\n{json.dumps(brief, indent=2, default=str)}\n\n"
        f"PLANNER:\n{json.dumps(asdict(planner), indent=2, default=str)}"
    )
    try:
        payload = _openai_reasoner(prompt, model=model)
        priority_tags = _as_str_list(payload.get("priority_tags", planner.priority_tags))
        suppress_tags = _as_str_list(payload.get("suppress_tags", planner.suppress_tags))
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
            rationale=_as_str_list(payload.get("rationale", [])),
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
        f"- active phase: `{state.active_phase or '-'}`",
        f"- active checklist item: `{state.active_checklist_item or '-'}`",
        f"- planner bottleneck: `{state.planner_bottleneck or '-'}`",
        f"- planner objective: `{state.planner_objective or '-'}`",
        f"- planner decision path: `{state.planner_decision_path or '-'}`",
        f"- critic decision path: `{state.critic_decision_path or '-'}`",
        f"- proposal path: `{state.proposal_path or '-'}`",
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


def _active_checklist_item_for_phase(
    phase: str,
    *,
    planner_decision: PlannerDecision | None = None,
) -> str:
    if phase == "pre_run_brief":
        return (
            "Control Plane / Split planning from heavy memory rebuild so pre-run control "
            "never blocks on full backfill."
        )
    if phase == "launching_bounded_loop":
        if (
            planner_decision is not None
            and planner_decision.active_bottleneck == "joint-metric-collapse"
        ):
            return (
                "Multitask Reset / Add staged multitask curriculum: detection-only control, "
                "frozen-trunk lane, then joint finetune."
            )
        return "Run Queue / Launch the next validated detection run from the fixed control plane."
    if phase == "post_run_sync":
        return (
            "Memory And State / Verify that the pre-run brief cites the real incumbent "
            "and active branch."
        )
    if phase == "waiting_external_run":
        return (
            "Run Queue / Relaunch the frontier supervisor after the memory upgrade so "
            "it plans from stronger retrieval."
        )
    return "Run Queue / Keep the fixed control plane alive with a queued validated successor."


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
    proposal_path: str | Path | None = DEFAULT_PROPOSAL_PATH,
) -> dict[str, Any]:
    """Run a continuous bounded research supervisor with memory + report publishing."""

    ensure_research_loop_enabled()
    dataset_root = Path(dataroot).expanduser().resolve()
    supervisor_root = Path(artifact_dir).expanduser()
    if not supervisor_root.is_absolute():
        supervisor_root = REPO_ROOT / supervisor_root
    supervisor_root = supervisor_root.resolve()
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
    pre_run_sync_enabled = _env_bool("TSQBEV_SUPERVISOR_PRE_RUN_SYNC", False)
    run_on_reject = _env_bool("TSQBEV_SUPERVISOR_RUN_ON_REJECT", True)
    memory_cfg = ResearchMemoryConfig.from_env()
    resolved_proposal_path = None
    if proposal_path is not None:
        resolved_proposal_path = Path(proposal_path).expanduser()
        if not resolved_proposal_path.is_absolute():
            resolved_proposal_path = REPO_ROOT / resolved_proposal_path
        resolved_proposal_path = resolved_proposal_path.resolve()

    while max_invocations is None or attempted_invocations < max_invocations:
        print("[supervisor] heartbeat: checking memory health", flush=True)
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
                active_phase="stopped",
                active_checklist_item=_active_checklist_item_for_phase("stopped"),
                proposal_path=str(resolved_proposal_path) if resolved_proposal_path else None,
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
                active_phase="waiting_external_run",
                active_checklist_item=_active_checklist_item_for_phase("waiting_external_run"),
                proposal_path=str(resolved_proposal_path) if resolved_proposal_path else None,
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
        pre_run_notes = [
            f"starting invocation `{invocation_root.name}`",
            (
                "using existing memory catalog for pre-run planning"
                if not pre_run_sync_enabled
                else "performing explicit pre-run memory sync"
            ),
        ]
        pre_run_state = SupervisorState(
            status="pre_run_brief",
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
            notes=pre_run_notes,
            active_phase="pre_run_brief",
            active_checklist_item=_active_checklist_item_for_phase("pre_run_brief"),
            proposal_path=str(resolved_proposal_path) if resolved_proposal_path else None,
        )
        _write_supervisor_outputs(
            pre_run_state,
            artifact_root=supervisor_root,
            report_path=DEFAULT_SUPERVISOR_REPORT,
        )
        print(
            f"[supervisor] invocation {attempted_invocations}: "
            f"pre-run sync {'on' if pre_run_sync_enabled else 'off'}",
            flush=True,
        )
        if pre_run_sync_enabled or not memory_cfg.catalog_path.exists():
            safe_sync_research_memory(REPO_ROOT, config=memory_cfg)
            print(
                f"[supervisor] invocation {attempted_invocations}: memory sync complete",
                flush=True,
            )
        pre_run_brief = safe_build_research_brief(REPO_ROOT, persist_log=False)
        print(
            f"[supervisor] invocation {attempted_invocations}: built pre-run brief",
            flush=True,
        )
        checkpoint_path = _write_first_principles_checkpoint(invocation_root, pre_run_brief)
        proposal_context = _load_proposal_context(resolved_proposal_path)
        planner_decision = _planner_decision_from_brief(
            pre_run_brief,
            proposal_context=proposal_context,
            proposal_path=resolved_proposal_path,
            runtime_root=invocation_root / "harness_runtime",
        )
        critic_decision = _critic_decision_from_planner(
            pre_run_brief,
            planner_decision,
            proposal_context=proposal_context,
        )
        supervisor_proposal = _build_supervisor_proposal(
            planner_decision,
            critic_decision,
            invocation_root=invocation_root,
        )
        (invocation_root / "proposal.json").write_text(
            json.dumps(supervisor_proposal.to_dict(), indent=2),
            encoding="utf-8",
        )
        _write_context_refresh_summary(
            invocation_root,
            phase="pre_run",
            proposal_path=resolved_proposal_path,
            proposal_context=proposal_context,
            brief=pre_run_brief,
            planner_decision=planner_decision,
            critic_decision=critic_decision,
            notes=pre_run_notes,
            summary=None,
        )
        print(
            f"[supervisor] invocation {attempted_invocations}: planner={planner_decision.provider} "
            f"critic={critic_decision.provider} approved={critic_decision.approved}",
            flush=True,
        )
        planner_provider = planner_decision.provider
        critic_provider = critic_decision.provider
        planner_decision_path = invocation_root / "planner_decision.json"
        critic_decision_path = invocation_root / "critic_decision.json"
        planner_decision_path.write_text(
            json.dumps(asdict(planner_decision), indent=2)
        )
        critic_decision_path.write_text(
            json.dumps(asdict(critic_decision), indent=2)
        )
        launch_state = SupervisorState(
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
            notes=[
                "planner provider "
                f"`{planner_decision.provider}` selected bottleneck "
                f"`{planner_decision.active_bottleneck}`",
                f"critic provider `{critic_decision.provider}` approved={critic_decision.approved}",
            ],
            active_phase="launching_bounded_loop",
            active_checklist_item=_active_checklist_item_for_phase(
                "launching_bounded_loop",
                planner_decision=planner_decision,
            ),
            planner_bottleneck=planner_decision.active_bottleneck,
            planner_objective=planner_decision.objective,
            planner_decision_path=str(planner_decision_path),
            critic_decision_path=str(critic_decision_path),
            proposal_path=str(resolved_proposal_path) if resolved_proposal_path else None,
        )
        _write_supervisor_outputs(
            launch_state,
            artifact_root=supervisor_root,
            report_path=DEFAULT_SUPERVISOR_REPORT,
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
        if not critic_decision.approved and not run_on_reject:
            notes.append("critic rejected the planner proposal; skipping execution")
            invocation_status = "rejected"
            summary = {
                "status": "rejected",
                "planner_decision": asdict(planner_decision),
                "critic_decision": asdict(critic_decision),
                "proposal": supervisor_proposal.to_dict(),
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
                "proposal": supervisor_proposal.to_dict(),
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
                active_phase="launching_bounded_loop",
                active_checklist_item=_active_checklist_item_for_phase(
                    "launching_bounded_loop",
                    planner_decision=planner_decision,
                ),
                planner_bottleneck=planner_decision.active_bottleneck,
                planner_objective=planner_decision.objective,
                planner_decision_path=str(planner_decision_path),
                critic_decision_path=str(critic_decision_path),
                proposal_path=str(resolved_proposal_path) if resolved_proposal_path else None,
            )
            _write_supervisor_outputs(
                state,
                artifact_root=supervisor_root,
                report_path=DEFAULT_SUPERVISOR_REPORT,
            )
            time.sleep(sleep_seconds)
            continue
        if not critic_decision.approved and run_on_reject:
            notes.append(
                "critic rejected the planner proposal, but fallback execution is enabled; "
                "running the bounded loop with the critic policy instead of idling the GPU"
            )
            invocation_status = "fallback_execution"
            print(
                f"[supervisor] invocation {attempted_invocations}: critic rejected proposal; "
                "executing fallback policy",
                flush=True,
            )
        try:
            print(
                f"[supervisor] invocation {attempted_invocations}: launching bounded loop",
                flush=True,
            )
            summary = run_bounded_research_loop(
                dataroot=dataset_root,
                artifact_dir=invocation_root,
                device=device,
                max_experiments=max_experiments,
                teacher_provider_config=teacher_provider_config,
                proposal=supervisor_proposal,
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
            print(
                f"[supervisor] invocation {attempted_invocations}: bounded loop complete",
                flush=True,
            )
        except Exception as exc:
            summary = {
                "status": "crash",
                "error": repr(exc),
                "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            }
            invocation_status = "crash"
            notes.append(f"invocation crashed: `{exc!r}`")
            print(
                f"[supervisor] invocation {attempted_invocations}: crash {exc!r}",
                flush=True,
            )

        _write_context_refresh_summary(
            invocation_root,
            phase="post_run",
            proposal_path=resolved_proposal_path,
            proposal_context=proposal_context,
            brief=pre_run_brief,
            planner_decision=planner_decision,
            critic_decision=critic_decision,
            notes=notes,
            summary=summary if isinstance(summary, dict) else {"summary": summary},
        )
        sync_timeout_seconds = int(
            os.getenv("TSQBEV_SUPERVISOR_SYNC_TIMEOUT_SECONDS", DEFAULT_SYNC_TIMEOUT_SECONDS)
        )
        brief_timeout_seconds = int(
            os.getenv("TSQBEV_SUPERVISOR_BRIEF_TIMEOUT_SECONDS", DEFAULT_BRIEF_TIMEOUT_SECONDS)
        )
        pre_publish_state = SupervisorState(
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
            notes=notes + ["starting bounded post-run maintenance"],
            active_phase="post_run_sync",
            active_checklist_item=_active_checklist_item_for_phase("post_run_sync"),
            planner_bottleneck=planner_decision.active_bottleneck,
            planner_objective=planner_decision.objective,
            planner_decision_path=str(planner_decision_path),
            critic_decision_path=str(critic_decision_path),
            proposal_path=str(resolved_proposal_path) if resolved_proposal_path else None,
        )
        _write_supervisor_outputs(
            pre_publish_state,
            artifact_root=supervisor_root,
            report_path=DEFAULT_SUPERVISOR_REPORT,
        )
        sync_result = _run_maintenance_cli(
            ["research-sync"],
            timeout_seconds=sync_timeout_seconds,
        )
        brief_result = _run_maintenance_cli(
            ["research-brief"],
            timeout_seconds=brief_timeout_seconds,
        )
        notes.append(
            "post-run memory sync status="
            f"`{sync_result['status']}` duration_s=`{sync_result.get('duration_s', 0.0):.1f}`"
        )
        notes.append(
            "post-run brief rebuild status="
            f"`{brief_result['status']}` duration_s=`{brief_result.get('duration_s', 0.0):.1f}`"
        )
        print(
            f"[supervisor] invocation {attempted_invocations}: post-run maintenance "
            f"sync={sync_result['status']} brief={brief_result['status']}",
            flush=True,
        )

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
            "maintenance": {
                "sync": sync_result,
                "brief": brief_result,
            },
            "summary": summary,
            "planner_decision": asdict(planner_decision),
            "critic_decision": asdict(critic_decision),
            "proposal": supervisor_proposal.to_dict(),
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
            active_phase="post_run_sync",
            active_checklist_item=_active_checklist_item_for_phase("post_run_sync"),
            planner_bottleneck=planner_decision.active_bottleneck,
            planner_objective=planner_decision.objective,
            planner_decision_path=str(planner_decision_path),
            critic_decision_path=str(critic_decision_path),
            proposal_path=str(resolved_proposal_path) if resolved_proposal_path else None,
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
