# Autoresearch Supervisor
_Generated: `2026-04-09T16:06:42.277766+00:00`_

## Status
- status: `running`
- branch: `main`
- repo sha: `03882d7`
- dataset root: `/mnt/storage/research/nuscenes`
- artifact root: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v6`
- attempted invocations: `1`
- completed invocations: `0`
- memory mode: `server`
- memory embedder: `fastembed`
- planner provider: `harness:proposal_aligned_v1`
- critic provider: `openai`
- active phase: `launching_bounded_loop`
- active checklist item: `Run Queue / Launch the next validated detection run from the fixed control plane.`
- planner bottleneck: `geometry-bridge-gap`
- planner objective: `replace the failing DINO-on-student path with a lightweight gated bridge student and teacher-side frontier supervision`
- planner decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v6/invocation_001_20260409-160630/planner_decision.json`
- critic decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v6/invocation_001_20260409-160630/critic_decision.json`
- proposal path: `-`
- last invocation dir: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v6/invocation_001_20260409-160630`
- last selected recipe: `-`
- last NDS: `-`
- last mAP: `-`
- last publish status: `-`
- last publish message: `-`

## Notes
- planner provider `harness:proposal_aligned_v1` selected bottleneck `geometry-bridge-gap`
- critic provider `openai` approved=True

## Pointers
- current PI brief: [docs/reports/current.md](docs/reports/current.md)
- supervisor ledger: [artifacts/autoresearch_frontier_v6/ledger.jsonl](artifacts/autoresearch_frontier_v6/ledger.jsonl)
- supervisor stop file: `artifacts/autoresearch_frontier_v6/STOP`
