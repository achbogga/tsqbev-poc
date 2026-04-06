from __future__ import annotations

from pathlib import Path

from tsqbev.research_supervisor import (
    SupervisorState,
    _active_checklist_item_for_phase,
    _as_str_list,
    _build_supervisor_proposal,
    _critic_decision_from_planner,
    _heuristic_planner,
    _load_proposal_context,
    _render_supervisor_report,
    _write_context_refresh_summary,
)


def test_heuristic_planner_detects_joint_metric_collapse() -> None:
    brief = {
        "current_state": [
            "joint_public_v3 collapsed at epoch 5 with NDS 0.0000 and mAP 0.0000",
        ]
    }
    decision = _heuristic_planner(brief)
    assert decision.active_bottleneck == "joint-metric-collapse"
    assert decision.force_priority_only is True
    assert "teacher_bag" in decision.suppress_tags


def test_critic_fallback_approves_low_token_burn_plan() -> None:
    brief = {"current_state": ["best frontier is v28 detection control"]}
    planner = _heuristic_planner(brief)
    critic = _critic_decision_from_planner(brief, planner)
    assert critic.approved is True
    assert critic.supervisor_policy["force_priority_only"] is True


def test_as_str_list_normalizes_scalar_strings() -> None:
    assert _as_str_list("reject this plan") == ["reject this plan"]
    assert _as_str_list(["a", 1]) == ["a", "1"]


def test_active_checklist_item_maps_joint_collapse_to_multitask_reset() -> None:
    planner = _heuristic_planner(
        {"current_state": ["joint_public_v3 collapsed with NDS 0.0000 and mAP 0.0000"]}
    )
    item = _active_checklist_item_for_phase("launching_bounded_loop", planner_decision=planner)
    assert "Multitask Reset" in item


def test_render_supervisor_report_includes_phase_and_planner_fields(tmp_path: Path) -> None:
    repo_tmp = Path("/home/achbogga/projects/tsqbev-poc/tests/.tmp")
    state = SupervisorState(
        status="running",
        generated_at_utc="2026-04-05T18:00:00+00:00",
        repo_sha="abc123",
        current_branch="main",
        dataset_root="/data",
        artifact_root="/artifacts",
        attempted_invocations=1,
        completed_invocations=0,
        last_invocation_dir="/artifacts/invocation_001",
        last_selected_recipe=None,
        last_nds=None,
        last_map=None,
        last_publish_status=None,
        last_publish_message=None,
        memory_mode="server",
        memory_embedder="fastembed",
        planner_provider="openai",
        critic_provider="openai",
        notes=["alive"],
        active_phase="launching_bounded_loop",
        active_checklist_item=(
            "Run Queue / Launch the next validated detection run from the fixed control plane."
        ),
        planner_bottleneck="quality-vs-calibration-boundary",
        planner_objective="improve nds",
        planner_decision_path="/artifacts/invocation_001/planner_decision.json",
        critic_decision_path="/artifacts/invocation_001/critic_decision.json",
        proposal_path="/docs/paper/tsqbev_frontier_program.md",
    )
    report = _render_supervisor_report(
        state,
        latest_brief_path=repo_tmp / "current.md",
        ledger_path=repo_tmp / "ledger.jsonl",
        stop_path=repo_tmp / "STOP",
    )
    assert "active phase" in report
    assert "planner bottleneck" in report
    assert "launching_bounded_loop" in report


def test_load_proposal_context_truncates_large_files(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal.md"
    proposal.write_text("# Proposal\n" + ("abc " * 5000))
    context = _load_proposal_context(proposal, max_chars=128)
    assert context.startswith("# Proposal")
    assert context.endswith("[truncated]")


def test_write_context_refresh_summary_writes_json_and_markdown(tmp_path: Path) -> None:
    planner = _heuristic_planner({"current_state": ["best frontier is v28 detection control"]})
    critic = _critic_decision_from_planner(
        {"current_state": ["best frontier is v28 detection control"]},
        planner,
        proposal_context="proposal thesis",
    )
    json_path, md_path = _write_context_refresh_summary(
        tmp_path,
        phase="post_run",
        proposal_path=None,
        proposal_context="proposal thesis",
        brief={
            "current_state": ["control is v29"],
            "open_blockers": ["scale-up blocked"],
            "recommended_next_steps": ["run DINOv3 + perspective supervision"],
            "evidence_refs": ["artifact summary"],
        },
        planner_decision=planner,
        critic_decision=critic,
        notes=["finished invocation"],
        summary={
            "selected_record": {
                "recipe": "frontier_run",
                "evaluation": {"nd_score": 0.2, "mean_ap": 0.19},
                "val": {"total": 10.5},
                "benchmark": {"mean_ms": 18.0},
            }
        },
    )
    assert json_path.exists()
    assert md_path.exists()
    assert "proposal thesis" in md_path.read_text()
    assert "frontier_run" in json_path.read_text()


def test_build_supervisor_proposal_uses_planner_and_critic_policy() -> None:
    planner = _heuristic_planner(
        {"current_state": ["joint_public_v3 collapsed with NDS 0.0000 and mAP 0.0000"]}
    )
    critic = _critic_decision_from_planner(
        {"current_state": ["joint_public_v3 collapsed with NDS 0.0000 and mAP 0.0000"]},
        planner,
        proposal_context="proposal thesis",
    )
    proposal = _build_supervisor_proposal(
        planner,
        critic,
        invocation_root=Path("/tmp/invocation_001"),
    )
    assert proposal.proposal_id == "invocation_001"
    assert proposal.targeted_bottleneck == "joint-metric-collapse"
    assert "teacher_off_control" in proposal.exploitation_tags
