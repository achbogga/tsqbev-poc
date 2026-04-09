# Autoresearch Supervisor
_Generated: `2026-04-09T16:45:54.248818+00:00`_

## Status
- status: `running`
- branch: `main`
- repo sha: `1c480a7`
- dataset root: `/mnt/storage/research/nuscenes`
- artifact root: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v6`
- attempted invocations: `2`
- completed invocations: `1`
- memory mode: `server`
- memory embedder: `fastembed`
- planner provider: `harness:proposal_aligned_v1`
- critic provider: `openai`
- active phase: `launching_bounded_loop`
- active checklist item: `Run Queue / Launch the next validated detection run from the fixed control plane.`
- planner bottleneck: `geometry-bridge-gap`
- planner objective: `replace the failing DINO-on-student path with a lightweight gated bridge student and teacher-side frontier supervision`
- planner decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v6/invocation_002_20260409-164543/planner_decision.json`
- critic decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v6/invocation_002_20260409-164543/critic_decision.json`
- proposal path: `-`
- last invocation dir: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v6/invocation_002_20260409-164543`
- last selected recipe: `None`
- last NDS: `0.0`
- last mAP: `0.0`
- last publish status: `published`
- last publish message: `autoresearch: publish invocation_001_20260409-160630 (2026-04-09 16:45 UTC)`

## Notes
- planner provider `harness:proposal_aligned_v1` selected bottleneck `geometry-bridge-gap`
- critic provider `openai` approved=True

## Pointers
- current PI brief: [docs/reports/current.md](docs/reports/current.md)
- supervisor ledger: [artifacts/autoresearch_frontier_v6/ledger.jsonl](artifacts/autoresearch_frontier_v6/ledger.jsonl)
- supervisor stop file: `artifacts/autoresearch_frontier_v6/STOP`
