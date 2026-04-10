"""Parallel meta-harness control plane for research orchestration.

References:
- Meta-Harnesses paper:
  https://arxiv.org/abs/2603.28052
- Karpathy autoresearch:
  https://github.com/karpathy/autoresearch
- Model Context Protocol:
  https://modelcontextprotocol.io/
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from tsqbev.research import _priority_tags_request_frontier
from tsqbev.research_memory import REPO_ROOT, safe_build_research_brief, safe_sync_research_memory

_OpenAI: Any
try:  # pragma: no cover - optional runtime dependency.
    from openai import OpenAI as _OpenAI
except ImportError:  # pragma: no cover
    _OpenAI = None
OpenAIClient: Any = _OpenAI

DEFAULT_HARNESS_ROOT = REPO_ROOT / "artifacts" / "harness_v2"
DEFAULT_HARNESS_REPORT = REPO_ROOT / "docs" / "reports" / "harness_v2.md"
DEFAULT_PROPOSAL_PATH = REPO_ROOT / "docs" / "paper" / "tsqbev_frontier_program.md"
DEFAULT_CONTEXT_BUDGET_CHARS = 16000
SUMMARY_THRESHOLD_RATIO = 0.50


@dataclass(slots=True)
class HarnessReplayTask:
    task_id: str
    title: str
    brief: dict[str, Any]
    proposal_context: str
    expected: dict[str, Any]
    category: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HarnessCandidateSpec:
    candidate_id: str
    source_path: Path
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_path": str(self.source_path),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class HarnessScorecard:
    candidate_id: str
    total_score: float
    decision_quality: float
    failure_diagnosis: float
    retrieval_quality: float
    execution_correctness: float
    efficiency: float
    publication_quality: float
    live_shadow_behavior: float
    failing_checks: list[str]
    per_task: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timestamp_tag() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _load_text(path: Path | None, *, max_chars: int = 12000) -> str:
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[truncated]"


def _brief_keywords(brief: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key in ("current_state", "open_blockers", "recommended_next_steps", "evidence_refs"):
        value = brief.get(key, [])
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, str):
                continue
            tokens.extend(re.findall(r"[a-zA-Z0-9_+.-]+", item.lower()))
    return tokens


def _estimate_context_chars(task: dict[str, Any]) -> int:
    return len(json.dumps(task, default=str))


def _summarize_brief(brief: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": brief.get("generated_at_utc"),
        "current_state": cast(list[str], brief.get("current_state", []))[:3],
        "open_blockers": cast(list[str], brief.get("open_blockers", []))[:3],
        "recommended_next_steps": cast(list[str], brief.get("recommended_next_steps", []))[:3],
        "evidence_refs": cast(list[str], brief.get("evidence_refs", []))[:4],
    }


def _persist_context_summary_if_needed(
    *,
    artifact_root: Path,
    phase: str,
    task: dict[str, Any],
    budget_chars: int,
) -> dict[str, Any] | None:
    used_chars = _estimate_context_chars(task)
    threshold_chars = int(budget_chars * SUMMARY_THRESHOLD_RATIO)
    if used_chars < threshold_chars:
        return None
    brief = task.get("brief", {})
    if not isinstance(brief, dict):
        brief = {}
    summary = {
        "kind": "harness_context_summary",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "phase": phase,
        "budget_chars": budget_chars,
        "used_chars": used_chars,
        "summary_threshold_chars": threshold_chars,
        "task_id": task.get("task_id"),
        "candidate_id": task.get("candidate_id"),
        "summary": _summarize_brief(brief),
    }
    summary_payload = cast(dict[str, Any], summary["summary"])
    _write_json(artifact_root / f"{phase}_context_summary.json", summary)
    _write_text(
        artifact_root / f"{phase}_context_summary.md",
        "\n".join(
            [
                f"# Harness Context Summary ({phase})",
                f"- generated_at_utc: `{summary['generated_at_utc']}`",
                f"- budget_chars: `{budget_chars}`",
                f"- used_chars: `{used_chars}`",
                "",
                "## Current State",
                *[f"- {line}" for line in cast(list[str], summary_payload["current_state"])],
                "",
                "## Open Blockers",
                *[f"- {line}" for line in cast(list[str], summary_payload["open_blockers"])],
                "",
                "## Recommended Next Steps",
                *[
                    f"- {line}"
                    for line in cast(list[str], summary_payload["recommended_next_steps"])
                ],
            ]
        )
        + "\n",
    )
    return summary


def _extract_python_block(text: str) -> str:
    fenced = re.search(r"```python\s+(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip() + "\n"
    fenced = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip() + "\n"
    return text.strip() + "\n"


def _openai_text(prompt: str, *, model: str) -> str:
    if OpenAIClient is None or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI client or key unavailable")
    client = OpenAIClient()
    response = client.responses.create(model=model, input=prompt)
    text = getattr(response, "output_text", "")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("OpenAI response did not return text output")
    return text


def _mcp_text(prompt: str, *, role: str) -> str:
    command = os.getenv("TSQBEV_HARNESS_MCP_CMD")
    if command is None:
        raise RuntimeError("TSQBEV_HARNESS_MCP_CMD is not configured")
    payload = {"role": role, "prompt": prompt}
    completed = subprocess.run(
        command,
        shell=True,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "MCP command failed")
    output = completed.stdout.strip()
    if not output:
        raise RuntimeError("MCP command returned empty output")
    return output


def _remote_text(
    prompt: str,
    *,
    role: str,
    provider: str,
    model: str | None = None,
) -> str:
    if provider == "mcp":
        return _mcp_text(prompt, role=role)
    if provider == "openai":
        resolved_model = model or (
            "gpt-5.4" if role in {"proposer", "judge"} else "gpt-5.4-mini"
        )
        return _openai_text(prompt, model=resolved_model)
    raise RuntimeError(f"unsupported remote provider: {provider}")


def _candidate_module_name(candidate_path: Path) -> str:
    digest = hashlib.sha1(str(candidate_path).encode("utf-8")).hexdigest()[:12]
    return f"tsqbev_harness_candidate_{digest}"


def _load_candidate(candidate_path: Path) -> HarnessCandidateSpec:
    module_name = _candidate_module_name(candidate_path)
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load candidate module from {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = getattr(module, "CANDIDATE_METADATA", {})
    run_fn = getattr(module, "run_harness", None)
    if not callable(run_fn):
        raise RuntimeError(f"{candidate_path} does not define run_harness(task)")
    candidate_id = str(metadata.get("candidate_id", candidate_path.parent.name))
    return HarnessCandidateSpec(
        candidate_id=candidate_id,
        source_path=candidate_path,
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _run_candidate(candidate: HarnessCandidateSpec, task: dict[str, Any]) -> dict[str, Any]:
    module_name = _candidate_module_name(candidate.source_path)
    spec = importlib.util.spec_from_file_location(module_name, candidate.source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load candidate module from {candidate.source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_harness(task)
    if not isinstance(result, dict):
        raise RuntimeError("candidate run_harness() must return a dict")
    return result


def _incumbent_candidate_source() -> str:
    return """from __future__ import annotations

CANDIDATE_METADATA = {
    "candidate_id": "incumbent_v1",
    "kind": "incumbent",
    "description": "Mirrors the current fixed supervisor heuristic planner.",
}


def run_harness(task: dict) -> dict:
    brief = task.get("brief", {})
    current_state = brief.get("current_state", [])
    state_text = "\\n".join(str(item) for item in current_state if isinstance(item, str)).lower()
    if "joint" in state_text and "0.0000" in state_text:
        return {
            "objective": (
                "stop broken joint promotion and focus on "
                "official-metric-safe detection control"
            ),
            "targeted_bottleneck": "joint-metric-collapse",
            "priority_tags": [
                "quality_rank_finegrid",
                "teacher_quality_plus",
                "teacher_off_control",
            ],
            "suppress_tags": [
                "teacher_bag",
                "anchor_mix",
                "augmentation",
                "lr_down",
                "query_boost",
            ],
            "force_priority_only": True,
            "kill_conditions": [
                "two consecutive runs fail to beat the current detection incumbent on NDS",
                "export sanity degrades while loss improves",
            ],
            "rationale": [
                "official joint detection collapsed to zero",
                "stay on the teacher-quality detection control line",
            ],
            "retrieval_queries": [
                "joint metric collapse zero nds detection control",
                "teacher quality plus quality rank incumbent",
            ],
            "worker_routes": [
                {"role": "planner", "executor": "remote"},
                {"role": "research_loop", "executor": "local_cli", "command": "research-loop"},
            ],
            "report_outline": ["Status", "Decision", "Metrics", "Next Steps"],
            "context_summary": "Use the current detection control line; avoid multitask churn.",
        }
    return {
        "objective": "improve official mini-val NDS without reopening dead exploit families",
        "targeted_bottleneck": "quality-vs-calibration-boundary",
        "priority_tags": ["quality_rank_finegrid", "teacher_quality_plus", "teacher_off_control"],
        "suppress_tags": ["teacher_bag", "anchor_mix", "augmentation", "lr_down"],
        "force_priority_only": True,
        "kill_conditions": [
            "two consecutive runs do not produce a meaningful progress verdict",
            "calibration improvements stop moving official NDS",
        ],
        "rationale": [
            "quality-rank and teacher-quality-plus are the only winning branches so far",
            "bag-style exploits and broad mutation fanout have underperformed",
        ],
        "retrieval_queries": [
            "current incumbent scale blocker next steps",
            "quality rank teacher quality frontier",
        ],
        "worker_routes": [
            {"role": "planner", "executor": "remote"},
            {"role": "research_loop", "executor": "local_cli", "command": "research-loop"},
        ],
        "report_outline": ["Status", "Decision", "Metrics", "Next Steps"],
        "context_summary": "Optimize the current local detection line with bounded experiments.",
    }
"""


def _proposal_aligned_candidate_source() -> str:
    return """from __future__ import annotations

CANDIDATE_METADATA = {
    "candidate_id": "proposal_aligned_v1",
    "kind": "frontier_bootstrap",
    "description": (
        "Proposal-aligned harness that replaces the custom student with a public "
        "working student."
    ),
}


def run_harness(task: dict) -> dict:
    brief = task.get("brief", {})
    proposal_context = str(task.get("proposal_context", "")).lower()
    current_state = brief.get("current_state", [])
    state_text = "\\n".join(str(item) for item in current_state if isinstance(item, str)).lower()
    blockers = "\\n".join(
        str(item) for item in brief.get("open_blockers", []) if isinstance(item, str)
    ).lower()
    catastrophic = "0.0000" in state_text or "sanity" in blockers or "geometry" in blockers
    joint_collapse = "joint" in state_text and "0.0000" in state_text
    frontier_tags = [
        "public_student_replacement",
        "bevdet_public_student",
        "camera_bev_working_baseline",
        "official_box_coder",
        "bevdepth_temporal_student",
        "geometry_sanity",
        "official_metric_only",
    ]
    suppress = [
        "quality_rank_finegrid",
        "teacher_quality_plus",
        "teacher_off_control",
        "teacher_bag",
        "anchor_mix",
        "augmentation",
        "lr_down",
        "schedule_only",
        "incremental_progress",
        "calibration_boundary",
        "lightweight_bridge",
        "gated_cross_attention",
        "teacher_side_foundation",
        "dino_v3",
        "dino_v3_bridge",
        "sam21_offline_support",
        "world_aligned_distillation",
        "world_latent_distillation",
        "sparse4d_efficiency",
    ]
    bottleneck = "custom-student-contract-failure"
    objective = "replace the broken custom student with a public working BEVDet student"
    if joint_collapse:
        bottleneck = "joint-metric-collapse"
        objective = "stop multitask collapse and restore metric-safe detection control"
        frontier_tags = frontier_tags + ["public_student_replacement"]
    elif catastrophic:
        bottleneck = "catastrophic-custom-student-failure"
        objective = "halt custom-student failures and move to a public working detector contract"
        frontier_tags = [
            "geometry_sanity",
            "official_metric_only",
        ] + frontier_tags
    elif (
        "world latent" in proposal_context
        or "dino" in proposal_context
        or "public student" in proposal_context
    ):
        bottleneck = "public-student-replacement"
        objective = (
            "replace the failing custom student with BEVDet first, then layer temporal "
            "depth upgrades only after nonzero official metrics are reproduced"
        )
    return {
        "objective": objective,
        "targeted_bottleneck": bottleneck,
        "priority_tags": frontier_tags,
        "suppress_tags": suppress,
        "force_priority_only": True,
        "kill_conditions": [
            "any official eval returns zero metrics or export sanity fails",
            "two consecutive public student runs fail to establish a nonzero official baseline",
            "the executor launches a custom-student branch under a public-student thesis",
        ],
        "rationale": [
            (
                "the custom TSQBEV student head/decoder contract is the current blocker, "
                "not the lack of another backbone trick"
            ),
            (
                "public working detector contracts should replace custom students before "
                "more research mutations are considered"
            ),
        ],
        "retrieval_queries": [
            "public student replacement BEVDet BEVDepth official box coder",
            "current incumbent scale blocker custom student head decoder",
            "repeated rabbit hole incremental progress schedule drift",
        ],
        "worker_routes": [
            {"role": "proposer", "executor": "remote_mcp_preferred"},
            {"role": "critic", "executor": "remote_mcp_preferred"},
            {"role": "benchmark", "executor": "local_cli", "command": "harness-benchmark"},
            {"role": "research_loop", "executor": "local_cli", "command": "research-loop"},
        ],
        "report_outline": [
            "Status",
            "Failure Signatures",
            "Public Student Thesis",
            "Executable Next Steps",
            "Kill Conditions",
        ],
        "context_summary": (
            "Prefer public working student replacement, kill catastrophic custom-student "
            "failures early, and suppress repeated low-ROI custom mutations."
        ),
    }
"""


def _candidate_root(root: Path, candidate_id: str) -> Path:
    return root / "candidates" / candidate_id


def _bootstrap_candidate(
    root: Path,
    *,
    candidate_id: str,
    source: str,
    metadata: dict[str, Any],
) -> Path:
    candidate_root = _candidate_root(root, candidate_id)
    candidate_root.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_root / "candidate.py"
    if not candidate_path.exists():
        _write_text(candidate_path, source)
    metadata_path = candidate_root / "metadata.json"
    merged_metadata = {"candidate_id": candidate_id, **metadata}
    _write_json(metadata_path, merged_metadata)
    return candidate_path


def _default_replay_tasks(
    *,
    brief: dict[str, Any],
    proposal_context: str,
) -> list[HarnessReplayTask]:
    live_expected = {
        "required_priority_tags": [
            "dino_v3_bridge",
            "bevformer_v2_perspective_supervision",
            "world_latent_distillation",
        ],
        "required_suppress_tags": ["quality_rank_finegrid", "teacher_quality_plus"],
        "required_queries": ["scale blocker", "bevfusion baseline", "incremental progress"],
        "required_outline": ["Status", "Next Steps"],
    }
    joint_brief = {
        "current_state": [
            "joint_public_v3 collapsed at epoch 5 with NDS 0.0000 and mAP 0.0000",
            "Detection export failed while lane loss looked acceptable.",
        ],
        "open_blockers": [
            "joint multitask setup is invalid until it passes official detection metrics"
        ],
        "recommended_next_steps": [
            "restore detection-only control and stage multitask integration later"
        ],
        "evidence_refs": ["joint_public_v3 log"],
    }
    catastrophic_brief = {
        "current_state": [
            "frontier run hit NDS 0.0000 and mAP 0.0000 on official eval",
            "export sanity failed with kilometer-scale boxes and saturated scores",
        ],
        "open_blockers": [
            "catastrophic geometry failure",
            "official metrics are invalid until export sanity is restored",
        ],
        "recommended_next_steps": [
            "stop the run immediately and inspect geometry/export bridge"
        ],
        "evidence_refs": ["metrics_summary.json", "export diagnostics"],
    }
    return [
        HarnessReplayTask(
            task_id="live_frontier_alignment",
            title="Frontier thesis alignment on the current repo brief",
            brief=brief,
            proposal_context=proposal_context,
            expected=live_expected,
            category="decision",
            weight=1.3,
        ),
        HarnessReplayTask(
            task_id="joint_metric_collapse",
            title="Detect and redirect the broken joint-training path",
            brief=joint_brief,
            proposal_context=proposal_context,
            expected={
                "required_bottleneck": "joint-metric-collapse",
                "required_priority_tags": ["overfit_gate_32_sample", "geometry_sanity"],
                "required_suppress_tags": ["teacher_bag", "anchor_mix", "quality_rank_finegrid"],
                "required_kill_terms": ["zero metrics", "legacy branch"],
                "required_outline": ["Failure Signatures", "Next Steps"],
            },
            category="failure",
            weight=1.0,
        ),
        HarnessReplayTask(
            task_id="catastrophic_geometry_failure",
            title="Hard-stop catastrophic official-eval collapse",
            brief=catastrophic_brief,
            proposal_context=proposal_context,
            expected={
                "required_bottleneck": "catastrophic-geometry-failure",
                "required_priority_tags": ["geometry_sanity", "official_metric_only"],
                "required_kill_terms": ["official eval", "export sanity"],
                "required_outline": ["Failure Signatures", "Kill Conditions"],
            },
            category="failure",
            weight=1.1,
        ),
        HarnessReplayTask(
            task_id="publication_quality",
            title="Generate a whole-cycle report outline with evidence and next steps",
            brief=brief,
            proposal_context=proposal_context,
            expected={
                "required_queries": ["incumbent", "scale blocker", "upstream baseline"],
                "required_outline": ["Status", "Failure Signatures", "Executable Next Steps"],
            },
            category="publication",
            weight=0.8,
        ),
    ]


def _normalize_strings(values: list[Any]) -> list[str]:
    return [str(value) for value in values if str(value).strip()]


def _score_task(
    result: dict[str, Any],
    task: HarnessReplayTask,
    *,
    incumbent_result: dict[str, Any] | None,
) -> dict[str, Any]:
    expected = task.expected
    priority_tags = _normalize_strings(cast(list[Any], result.get("priority_tags", [])))
    suppress_tags = _normalize_strings(cast(list[Any], result.get("suppress_tags", [])))
    kill_conditions = " ".join(
        _normalize_strings(cast(list[Any], result.get("kill_conditions", [])))
    ).lower()
    retrieval_queries = " ".join(
        _normalize_strings(cast(list[Any], result.get("retrieval_queries", [])))
    ).lower()
    report_outline = _normalize_strings(cast(list[Any], result.get("report_outline", [])))
    objective = str(result.get("objective", ""))
    bottleneck = str(result.get("targeted_bottleneck", ""))

    decision_score = 0.0
    failure_score = 0.0
    retrieval_score = 0.0
    execution_score = 0.0
    efficiency_score = 0.0
    publication_score = 0.0
    shadow_score = 0.0
    failing_checks: list[str] = []

    required_bottleneck = expected.get("required_bottleneck")
    if required_bottleneck is None or bottleneck == required_bottleneck:
        decision_score += 1.0
    else:
        failing_checks.append(f"bottleneck:{required_bottleneck}")

    required_priority_tags = cast(list[str], expected.get("required_priority_tags", []))
    hits = sum(1 for tag in required_priority_tags if tag in priority_tags)
    if required_priority_tags:
        decision_score += hits / len(required_priority_tags)
    required_suppress_tags = cast(list[str], expected.get("required_suppress_tags", []))
    suppress_hits = sum(1 for tag in required_suppress_tags if tag in suppress_tags)
    if required_suppress_tags:
        failure_score += suppress_hits / len(required_suppress_tags)
    required_kill_terms = cast(list[str], expected.get("required_kill_terms", []))
    kill_hits = sum(1 for term in required_kill_terms if term in kill_conditions)
    if required_kill_terms:
        failure_score += kill_hits / len(required_kill_terms)
    required_queries = cast(list[str], expected.get("required_queries", []))
    query_hits = sum(1 for term in required_queries if term.lower() in retrieval_queries)
    if required_queries:
        retrieval_score += query_hits / len(required_queries)
    required_outline = cast(list[str], expected.get("required_outline", []))
    outline_hits = sum(1 for item in required_outline if item in report_outline)
    if required_outline:
        publication_score += outline_hits / len(required_outline)

    if priority_tags:
        if _priority_tags_request_frontier(priority_tags):
            execution_score += 1.0
        elif "frontier" in task.task_id or "catastrophic" in task.task_id:
            failing_checks.append("execution:not_frontier")
    if str(result.get("force_priority_only", False)).lower() in {"true", "1"}:
        execution_score += 0.25

    context_chars = _estimate_context_chars(result)
    if context_chars <= DEFAULT_CONTEXT_BUDGET_CHARS:
        efficiency_score = 1.0
    elif context_chars <= int(DEFAULT_CONTEXT_BUDGET_CHARS * 1.2):
        efficiency_score = 0.5
    else:
        failing_checks.append("efficiency:context_budget")

    if incumbent_result is not None and task.task_id == "live_frontier_alignment":
        incumbent_priority = _normalize_strings(
            cast(list[Any], incumbent_result.get("priority_tags", []))
        )
        if hits > sum(1 for tag in required_priority_tags if tag in incumbent_priority):
            shadow_score = 1.0
        elif priority_tags == incumbent_priority:
            shadow_score = 0.5
        else:
            failing_checks.append("shadow:no_gain")

    decision_score = min(decision_score, 1.0)
    failure_score = min(failure_score, 1.0)
    retrieval_score = min(retrieval_score, 1.0)
    execution_score = min(execution_score, 1.0)
    efficiency_score = min(efficiency_score, 1.0)
    publication_score = min(publication_score, 1.0)
    shadow_score = min(shadow_score, 1.0)

    return {
        "task_id": task.task_id,
        "title": task.title,
        "category": task.category,
        "weight": task.weight,
        "decision_quality": decision_score,
        "failure_diagnosis": failure_score,
        "retrieval_quality": retrieval_score,
        "execution_correctness": execution_score,
        "efficiency": efficiency_score,
        "publication_quality": publication_score,
        "live_shadow_behavior": shadow_score,
        "objective": objective,
        "targeted_bottleneck": bottleneck,
        "failing_checks": failing_checks,
    }


def _aggregate_scorecard(
    *,
    candidate_id: str,
    per_task: list[dict[str, Any]],
) -> HarnessScorecard:
    def weighted_total(key: str, scale: float) -> float:
        numerator = 0.0
        denominator = 0.0
        for item in per_task:
            weight = float(item["weight"])
            numerator += float(item[key]) * weight
            denominator += weight
        if denominator == 0:
            return 0.0
        return round(scale * numerator / denominator, 3)

    decision_quality = weighted_total("decision_quality", 25.0)
    failure_diagnosis = weighted_total("failure_diagnosis", 20.0)
    retrieval_quality = weighted_total("retrieval_quality", 15.0)
    execution_correctness = weighted_total("execution_correctness", 15.0)
    efficiency = weighted_total("efficiency", 10.0)
    publication_quality = weighted_total("publication_quality", 10.0)
    live_shadow_behavior = weighted_total("live_shadow_behavior", 5.0)
    total_score = round(
        decision_quality
        + failure_diagnosis
        + retrieval_quality
        + execution_correctness
        + efficiency
        + publication_quality
        + live_shadow_behavior,
        3,
    )
    failing_checks = sorted(
        {
            check
            for item in per_task
            for check in cast(list[str], item.get("failing_checks", []))
        }
    )
    return HarnessScorecard(
        candidate_id=candidate_id,
        total_score=total_score,
        decision_quality=decision_quality,
        failure_diagnosis=failure_diagnosis,
        retrieval_quality=retrieval_quality,
        execution_correctness=execution_correctness,
        efficiency=efficiency,
        publication_quality=publication_quality,
        live_shadow_behavior=live_shadow_behavior,
        failing_checks=failing_checks,
        per_task=per_task,
    )


def _benchmark_summary_payload(
    *,
    spec: HarnessCandidateSpec,
    scorecard: HarnessScorecard,
    suite_path: Path,
) -> dict[str, Any]:
    return {
        "kind": "harness_benchmark",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "candidate_id": spec.candidate_id,
        "candidate_path": str(spec.source_path),
        "scorecard": scorecard.to_dict(),
        "suite_path": str(suite_path),
        "status": "completed",
        "selected_recipe": spec.candidate_id,
    }


def _shadow_summary_payload(
    *,
    candidate_id: str,
    incumbent_id: str,
    candidate_score: HarnessScorecard,
    incumbent_score: HarnessScorecard,
) -> dict[str, Any]:
    score_gap = round(candidate_score.total_score - incumbent_score.total_score, 3)
    return {
        "kind": "harness_shadow",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "candidate_id": candidate_id,
        "incumbent_id": incumbent_id,
        "candidate_total_score": candidate_score.total_score,
        "incumbent_total_score": incumbent_score.total_score,
        "score_gap": score_gap,
        "status": "promotable" if score_gap >= 8.0 else "shadow_only",
        "selected_recipe": candidate_id,
    }


def _promotion_summary_payload(
    *,
    candidate_id: str,
    benchmark: HarnessScorecard,
    incumbent: HarnessScorecard,
    shadow: dict[str, Any],
    promoted: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "kind": "harness_promotion",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "candidate_id": candidate_id,
        "benchmark_total_score": benchmark.total_score,
        "incumbent_total_score": incumbent.total_score,
        "shadow_score_gap": shadow.get("score_gap"),
        "status": "promoted" if promoted else "rejected",
        "reason": reason,
        "selected_recipe": candidate_id,
    }


def _default_provider() -> str:
    if os.getenv("TSQBEV_HARNESS_MCP_CMD"):
        return "mcp"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "heuristic"


def _proposer_prompt(
    *,
    best_candidate_code: str,
    best_score: HarnessScorecard,
    brief: dict[str, Any],
    proposal_context: str,
) -> str:
    return (
        "Write a complete single-file Python harness candidate. Output only code.\n"
        "Contract:\n"
        "- define CANDIDATE_METADATA dict\n"
        "- define run_harness(task: dict) -> dict\n"
        "- return keys: objective, targeted_bottleneck, priority_tags, suppress_tags, "
        "force_priority_only, kill_conditions, rationale, retrieval_queries, "
        "worker_routes, report_outline, context_summary\n"
        "Use simple deterministic logic; no external imports beyond stdlib.\n\n"
        f"CURRENT_BEST_SCORECARD:\n{json.dumps(best_score.to_dict(), indent=2)}\n\n"
        f"CURRENT_BRIEF:\n{json.dumps(brief, indent=2)}\n\n"
        f"PROPOSAL_CONTEXT:\n{proposal_context}\n\n"
        f"CURRENT_BEST_CANDIDATE:\n```python\n{best_candidate_code}\n```"
    )


def _judge_prompt(*, scorecard: HarnessScorecard) -> str:
    return (
        "Review this harness scorecard and return JSON with keys: "
        "approved, rationale, next_focus. Be concise.\n\n"
        f"SCORECARD:\n{json.dumps(scorecard.to_dict(), indent=2)}"
    )


def _generate_candidate_source(
    *,
    provider: str,
    brief: dict[str, Any],
    proposal_context: str,
    best_candidate_source: str,
    best_score: HarnessScorecard,
) -> tuple[str, dict[str, Any]]:
    if provider == "heuristic":
        return (
            _proposal_aligned_candidate_source(),
            {
                "provider": "heuristic",
                "rationale": [
                    "used the built-in proposal-aligned candidate because no remote provider "
                    "was configured"
                ],
            },
        )
    proposer_text = _remote_text(
        _proposer_prompt(
            best_candidate_code=best_candidate_source,
            best_score=best_score,
            brief=brief,
            proposal_context=proposal_context,
        ),
        role="proposer",
        provider=provider,
    )
    judge_payload: dict[str, Any] = {
        "provider": provider,
        "rationale": ["remote proposer generated candidate source"],
    }
    candidate_source = _extract_python_block(proposer_text)
    try:
        judge_text = _remote_text(
            _judge_prompt(scorecard=best_score),
            role="judge",
            provider=provider,
        )
        judge_payload["judge_output"] = judge_text
    except Exception as exc:  # pragma: no cover - defensive
        judge_payload["judge_error"] = repr(exc)
    return candidate_source, judge_payload


def run_harness_benchmark(
    *,
    artifact_dir: str | Path = DEFAULT_HARNESS_ROOT,
    candidate_path: str | Path | None = None,
    proposal_path: str | Path | None = DEFAULT_PROPOSAL_PATH,
    budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
) -> dict[str, Any]:
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    brief = safe_build_research_brief(REPO_ROOT, persist_log=False)
    proposal_context = _load_text(Path(proposal_path) if proposal_path is not None else None)
    incumbent_path = _bootstrap_candidate(
        root,
        candidate_id="incumbent_v1",
        source=_incumbent_candidate_source(),
        metadata={"kind": "incumbent"},
    )
    if candidate_path is None:
        candidate_path = incumbent_path
    spec = _load_candidate(Path(candidate_path))
    incumbent_spec = _load_candidate(incumbent_path)
    suite = _default_replay_tasks(brief=brief, proposal_context=proposal_context)
    run_root = root / "benchmarks" / f"{spec.candidate_id}_{_timestamp_tag()}"
    run_root.mkdir(parents=True, exist_ok=True)
    suite_path = _write_json(
        run_root / "suite.json",
        {"tasks": [task.to_dict() for task in suite]},
    )
    incumbent_results: dict[str, dict[str, Any]] = {}
    if spec.candidate_id != incumbent_spec.candidate_id:
        for task in suite:
            incumbent_task = {
                "task_id": task.task_id,
                "title": task.title,
                "brief": task.brief,
                "proposal_context": task.proposal_context,
                "budget_chars": budget_chars,
                "candidate_id": incumbent_spec.candidate_id,
            }
            incumbent_results[task.task_id] = _run_candidate(incumbent_spec, incumbent_task)

    per_task: list[dict[str, Any]] = []
    for task in suite:
        candidate_task = {
            "task_id": task.task_id,
            "title": task.title,
            "brief": task.brief,
            "proposal_context": task.proposal_context,
            "budget_chars": budget_chars,
            "candidate_id": spec.candidate_id,
        }
        _persist_context_summary_if_needed(
            artifact_root=run_root / task.task_id,
            phase="benchmark",
            task=candidate_task,
            budget_chars=budget_chars,
        )
        result = _run_candidate(spec, candidate_task)
        _write_json(run_root / task.task_id / "result.json", result)
        per_task.append(
            _score_task(
                result,
                task,
                incumbent_result=incumbent_results.get(task.task_id),
            )
        )
    scorecard = _aggregate_scorecard(candidate_id=spec.candidate_id, per_task=per_task)
    summary = _benchmark_summary_payload(spec=spec, scorecard=scorecard, suite_path=suite_path)
    _write_json(run_root / "scorecard.json", scorecard.to_dict())
    _write_json(run_root / "summary.json", summary)
    _write_text(
        run_root / "report.md",
        "\n".join(
            [
                f"# Harness Benchmark: {spec.candidate_id}",
                f"- total_score: `{scorecard.total_score}`",
                f"- decision_quality: `{scorecard.decision_quality}`",
                f"- failure_diagnosis: `{scorecard.failure_diagnosis}`",
                f"- retrieval_quality: `{scorecard.retrieval_quality}`",
                f"- execution_correctness: `{scorecard.execution_correctness}`",
                f"- efficiency: `{scorecard.efficiency}`",
                f"- publication_quality: `{scorecard.publication_quality}`",
                f"- live_shadow_behavior: `{scorecard.live_shadow_behavior}`",
                "",
                "## Failing Checks",
                *[f"- {item}" for item in scorecard.failing_checks],
            ]
        )
        + "\n",
    )
    return {
        "candidate": spec.to_dict(),
        "benchmark_root": str(run_root),
        "scorecard": scorecard.to_dict(),
    }


def run_harness_shadow(
    *,
    artifact_dir: str | Path = DEFAULT_HARNESS_ROOT,
    candidate_path: str | Path,
    proposal_path: str | Path | None = DEFAULT_PROPOSAL_PATH,
    budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
) -> dict[str, Any]:
    root = Path(artifact_dir)
    brief = safe_build_research_brief(REPO_ROOT, persist_log=False)
    proposal_context = _load_text(Path(proposal_path) if proposal_path is not None else None)
    incumbent_path = _bootstrap_candidate(
        root,
        candidate_id="incumbent_v1",
        source=_incumbent_candidate_source(),
        metadata={"kind": "incumbent"},
    )
    candidate_spec = _load_candidate(Path(candidate_path))
    incumbent_spec = _load_candidate(incumbent_path)
    candidate_benchmark = run_harness_benchmark(
        artifact_dir=root,
        candidate_path=candidate_spec.source_path,
        proposal_path=proposal_path,
        budget_chars=budget_chars,
    )
    incumbent_benchmark = run_harness_benchmark(
        artifact_dir=root,
        candidate_path=incumbent_spec.source_path,
        proposal_path=proposal_path,
        budget_chars=budget_chars,
    )
    candidate_score = HarnessScorecard(**candidate_benchmark["scorecard"])
    incumbent_score = HarnessScorecard(**incumbent_benchmark["scorecard"])
    shadow_root = root / "shadow" / f"{candidate_spec.candidate_id}_{_timestamp_tag()}"
    shadow_root.mkdir(parents=True, exist_ok=True)
    _persist_context_summary_if_needed(
        artifact_root=shadow_root,
        phase="shadow",
        task={
            "task_id": "shadow_live_brief",
            "brief": brief,
            "proposal_context": proposal_context,
            "candidate_id": candidate_spec.candidate_id,
        },
        budget_chars=budget_chars,
    )
    summary = _shadow_summary_payload(
        candidate_id=candidate_spec.candidate_id,
        incumbent_id=incumbent_spec.candidate_id,
        candidate_score=candidate_score,
        incumbent_score=incumbent_score,
    )
    _write_json(shadow_root / "summary.json", summary)
    _write_text(
        shadow_root / "report.md",
        "\n".join(
            [
                f"# Harness Shadow: {candidate_spec.candidate_id}",
                f"- candidate_total_score: `{candidate_score.total_score}`",
                f"- incumbent_total_score: `{incumbent_score.total_score}`",
                f"- score_gap: `{summary['score_gap']}`",
                f"- status: `{summary['status']}`",
            ]
        )
        + "\n",
    )
    return {
        "shadow_root": str(shadow_root),
        "summary": summary,
        "candidate_scorecard": candidate_score.to_dict(),
        "incumbent_scorecard": incumbent_score.to_dict(),
    }


def run_harness_promote(
    *,
    artifact_dir: str | Path = DEFAULT_HARNESS_ROOT,
    candidate_path: str | Path,
    proposal_path: str | Path | None = DEFAULT_PROPOSAL_PATH,
    budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
) -> dict[str, Any]:
    root = Path(artifact_dir)
    candidate_spec = _load_candidate(Path(candidate_path))
    shadow = run_harness_shadow(
        artifact_dir=root,
        candidate_path=candidate_spec.source_path,
        proposal_path=proposal_path,
        budget_chars=budget_chars,
    )
    candidate_score = HarnessScorecard(**shadow["candidate_scorecard"])
    incumbent_score = HarnessScorecard(**shadow["incumbent_scorecard"])
    score_gap = float(shadow["summary"]["score_gap"])
    promoted = (
        score_gap >= 8.0
        and candidate_score.execution_correctness >= incumbent_score.execution_correctness
    )
    reason = (
        "candidate beat the incumbent by the required score margin"
        if promoted
        else "candidate did not clear the promotion gates"
    )
    promotion_root = root / "promotions" / f"{candidate_spec.candidate_id}_{_timestamp_tag()}"
    promotion_root.mkdir(parents=True, exist_ok=True)
    summary = _promotion_summary_payload(
        candidate_id=candidate_spec.candidate_id,
        benchmark=candidate_score,
        incumbent=incumbent_score,
        shadow=shadow["summary"],
        promoted=promoted,
        reason=reason,
    )
    _write_json(promotion_root / "summary.json", summary)
    if promoted:
        current_root = root / "promoted"
        current_root.mkdir(parents=True, exist_ok=True)
        current_candidate = current_root / "candidate.py"
        shutil.copy2(candidate_spec.source_path, current_candidate)
        _write_json(
            current_root / "current.json",
            {
                "candidate_id": candidate_spec.candidate_id,
                "source_path": str(candidate_spec.source_path),
                "promoted_at_utc": datetime.now(tz=UTC).isoformat(),
                "score_gap": score_gap,
            },
        )
    return {
        "promotion_root": str(promotion_root),
        "summary": summary,
    }


def load_promoted_harness_plan(
    *,
    brief: dict[str, Any],
    proposal_path: str | Path | None = DEFAULT_PROPOSAL_PATH,
    artifact_dir: str | Path = DEFAULT_HARNESS_ROOT,
    budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
    runtime_root: str | Path | None = None,
) -> dict[str, Any] | None:
    root = Path(artifact_dir)
    promoted_path = root / "promoted" / "current.json"
    if not promoted_path.exists():
        return None
    promoted = json.loads(promoted_path.read_text(encoding="utf-8"))
    current_candidate = root / "promoted" / "candidate.py"
    source_path = current_candidate if current_candidate.exists() else Path(
        str(promoted["source_path"])
    )
    spec = _load_candidate(source_path)
    runtime_dir = (
        Path(runtime_root)
        if runtime_root is not None
        else root / "runtime" / f"{spec.candidate_id}_{_timestamp_tag()}"
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    task = {
        "task_id": f"live_control_{_timestamp_tag()}",
        "candidate_id": spec.candidate_id,
        "title": "Live research control decision",
        "brief": brief,
        "proposal_context": _load_text(
            Path(proposal_path) if proposal_path is not None else None
        ),
    }
    summary = _persist_context_summary_if_needed(
        artifact_root=runtime_dir,
        phase="live_control",
        task=task,
        budget_chars=budget_chars,
    )
    plan = _run_candidate(spec, task)
    payload = {
        "candidate_id": spec.candidate_id,
        "candidate_path": str(spec.source_path),
        "runtime_root": str(runtime_dir),
        "plan": plan,
        "context_summary_path": (
            str(runtime_dir / "live_control_context_summary.json") if summary is not None else None
        ),
    }
    _write_json(runtime_dir / "plan.json", payload)
    return payload


def run_harness_search(
    *,
    artifact_dir: str | Path = DEFAULT_HARNESS_ROOT,
    proposal_path: str | Path | None = DEFAULT_PROPOSAL_PATH,
    iterations: int = 3,
    provider: str | None = None,
    budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
) -> dict[str, Any]:
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    brief = safe_build_research_brief(REPO_ROOT, persist_log=False)
    proposal_context = _load_text(Path(proposal_path) if proposal_path is not None else None)
    incumbent_path = _bootstrap_candidate(
        root,
        candidate_id="incumbent_v1",
        source=_incumbent_candidate_source(),
        metadata={"kind": "incumbent"},
    )
    _bootstrap_candidate(
        root,
        candidate_id="proposal_aligned_v1",
        source=_proposal_aligned_candidate_source(),
        metadata={"kind": "frontier_bootstrap"},
    )
    resolved_provider = provider or _default_provider()
    leaderboard: list[dict[str, Any]] = []
    incumbent_benchmark = run_harness_benchmark(
        artifact_dir=root,
        candidate_path=incumbent_path,
        proposal_path=proposal_path,
        budget_chars=budget_chars,
    )
    incumbent_score = HarnessScorecard(**incumbent_benchmark["scorecard"])
    best_candidate_path = incumbent_path
    best_score = incumbent_score
    leaderboard.append(
        {
            "candidate_id": "incumbent_v1",
            "candidate_path": str(incumbent_path),
            "total_score": incumbent_score.total_score,
            "provider": "bootstrap",
        }
    )

    search_root = root / "search" / f"run_{_timestamp_tag()}"
    search_root.mkdir(parents=True, exist_ok=True)
    for iteration in range(1, max(iterations, 1) + 1):
        if iteration == 1:
            candidate_source = _proposal_aligned_candidate_source()
            generation_meta = {
                "provider": "bootstrap",
                "rationale": ["seeded the search with a proposal-aligned frontier harness"],
            }
        else:
            candidate_source, generation_meta = _generate_candidate_source(
                provider=resolved_provider,
                brief=brief,
                proposal_context=proposal_context,
                best_candidate_source=best_candidate_path.read_text(encoding="utf-8"),
                best_score=best_score,
            )
        candidate_id = f"candidate_{iteration:03d}"
        candidate_root = _candidate_root(root, candidate_id)
        candidate_root.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_root / "candidate.py"
        _write_text(candidate_path, candidate_source)
        _write_json(candidate_root / "generation.json", generation_meta)
        benchmark = run_harness_benchmark(
            artifact_dir=root,
            candidate_path=candidate_path,
            proposal_path=proposal_path,
            budget_chars=budget_chars,
        )
        score = HarnessScorecard(**benchmark["scorecard"])
        leaderboard.append(
            {
                "candidate_id": candidate_id,
                "candidate_path": str(candidate_path),
                "total_score": score.total_score,
                "provider": generation_meta.get("provider", resolved_provider),
            }
        )
        if score.total_score > best_score.total_score:
            best_score = score
            best_candidate_path = candidate_path

    leaderboard = sorted(leaderboard, key=lambda item: float(item["total_score"]), reverse=True)
    _write_json(search_root / "leaderboard.json", {"leaderboard": leaderboard})
    report_lines = [
        "# Harness Search",
        f"- provider: `{resolved_provider}`",
        f"- best_candidate: `{best_candidate_path.parent.name}`",
        f"- best_score: `{best_score.total_score}`",
        "",
        "## Leaderboard",
    ]
    report_lines.extend(
        [
            f"- `{entry['candidate_id']}`: total_score `{entry['total_score']}` "
            f"(provider `{entry['provider']}`)"
            for entry in leaderboard
        ]
    )
    _write_text(search_root / "report.md", "\n".join(report_lines) + "\n")
    return {
        "search_root": str(search_root),
        "provider": resolved_provider,
        "best_candidate_path": str(best_candidate_path),
        "best_scorecard": best_score.to_dict(),
        "leaderboard": leaderboard,
    }


def render_harness_report(
    *,
    artifact_dir: str | Path = DEFAULT_HARNESS_ROOT,
    report_path: str | Path = DEFAULT_HARNESS_REPORT,
) -> dict[str, Any]:
    root = Path(artifact_dir)
    leaderboard_entries: list[dict[str, Any]] = []
    for path in sorted((root / "search").glob("run_*/leaderboard.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("leaderboard", []):
            if isinstance(item, dict):
                leaderboard_entries.append(item)
    promoted_path = root / "promoted" / "current.json"
    promoted = json.loads(promoted_path.read_text()) if promoted_path.exists() else None
    leaderboard_entries = sorted(
        leaderboard_entries,
        key=lambda item: float(item.get("total_score", 0.0)),
        reverse=True,
    )
    lines = [
        "# Harness V2",
        f"_Generated: `{datetime.now(tz=UTC).isoformat()}`_",
        "",
        "## Promoted",
    ]
    if promoted is None:
        lines.append("- no promoted harness candidate yet")
    else:
        lines.extend(
            [
                f"- candidate_id: `{promoted['candidate_id']}`",
                f"- score_gap: `{promoted['score_gap']}`",
                f"- source_path: `{promoted['source_path']}`",
            ]
        )
    lines.extend(["", "## Leaderboard"])
    if not leaderboard_entries:
        lines.append("- no harness search runs yet")
    else:
        lines.extend(
            [
                f"- `{item['candidate_id']}`: total_score `{item['total_score']}` "
                f"(provider `{item['provider']}`)"
                for item in leaderboard_entries[:10]
            ]
        )
    report = "\n".join(lines) + "\n"
    _write_text(Path(report_path), report)
    return {
        "report_path": str(report_path),
        "promoted": promoted,
        "leaderboard_count": len(leaderboard_entries),
    }


def sync_harness_memory(*, artifact_dir: str | Path = DEFAULT_HARNESS_ROOT) -> dict[str, Any]:
    result = safe_sync_research_memory(REPO_ROOT)
    render_harness_report(artifact_dir=artifact_dir)
    return result
