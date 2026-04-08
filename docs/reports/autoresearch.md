# Autoresearch Supervisor
_Generated: `2026-04-07T22:50:17.046892+00:00`_

## Status
- status: `running`
- branch: `main`
- repo sha: `941f0aa`
- dataset root: `/home/achbogga/projects/research/nuscenes`
- artifact root: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v4`
- attempted invocations: `5`
- completed invocations: `4`
- memory mode: `server`
- memory embedder: `fastembed`
- planner provider: `heuristic`
- critic provider: `openai`
- active phase: `launching_bounded_loop`
- active checklist item: `Run Queue / Launch the next validated detection run from the fixed control plane.`
- planner bottleneck: `quality-vs-calibration-boundary`
- planner objective: `improve official mini-val NDS without reopening dead exploit families`
- planner decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v4/invocation_005_20260407-225002/planner_decision.json`
- critic decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v4/invocation_005_20260407-225002/critic_decision.json`
- proposal path: `/home/achbogga/projects/tsqbev-poc/docs/paper/tsqbev_frontier_program.md`
- last invocation dir: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v4/invocation_005_20260407-225002`
- last selected recipe: `carryover_recovery_v14_teacher_anchor_quality_focal`
- last NDS: `0.13532017474895897`
- last mAP: `0.16814227206244053`
- last publish status: `published`
- last publish message: `autoresearch: publish invocation_004_20260407-213735 (2026-04-07 22:49 UTC)`

## Notes
- planner provider `heuristic` selected bottleneck `quality-vs-calibration-boundary`
- critic provider `openai` approved=False

## Pointers
- current PI brief: [docs/reports/current.md](docs/reports/current.md)
- supervisor ledger: [artifacts/autoresearch_frontier_v4/ledger.jsonl](artifacts/autoresearch_frontier_v4/ledger.jsonl)
- supervisor stop file: `artifacts/autoresearch_frontier_v4/STOP`
