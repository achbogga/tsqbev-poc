# Autoresearch Supervisor
_Generated: `2026-04-07T09:57:54.256581+00:00`_

## Status
- status: `running`
- branch: `main`
- repo sha: `e354f03`
- dataset root: `/home/achbogga/projects/research/nuscenes`
- artifact root: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v4`
- attempted invocations: `3`
- completed invocations: `2`
- memory mode: `server`
- memory embedder: `fastembed`
- planner provider: `heuristic`
- critic provider: `openai`
- active phase: `launching_bounded_loop`
- active checklist item: `Run Queue / Launch the next validated detection run from the fixed control plane.`
- planner bottleneck: `quality-vs-calibration-boundary`
- planner objective: `improve official mini-val NDS without reopening dead exploit families`
- planner decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v4/invocation_003_20260407-095738/planner_decision.json`
- critic decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v4/invocation_003_20260407-095738/critic_decision.json`
- proposal path: `/home/achbogga/projects/tsqbev-poc/docs/paper/tsqbev_frontier_program.md`
- last invocation dir: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v4/invocation_003_20260407-095738`
- last selected recipe: `None`
- last NDS: `0.0`
- last mAP: `0.0`
- last publish status: `published`
- last publish message: `autoresearch: publish invocation_002_20260406-223637 (2026-04-07 09:57 UTC)`

## Notes
- planner provider `heuristic` selected bottleneck `quality-vs-calibration-boundary`
- critic provider `openai` approved=False

## Pointers
- current PI brief: [docs/reports/current.md](docs/reports/current.md)
- supervisor ledger: [artifacts/autoresearch_frontier_v4/ledger.jsonl](artifacts/autoresearch_frontier_v4/ledger.jsonl)
- supervisor stop file: `artifacts/autoresearch_frontier_v4/STOP`
