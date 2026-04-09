# Autoresearch Supervisor
_Generated: `2026-04-09T18:16:58.514399+00:00`_

## Status
- status: `running`
- branch: `main`
- repo sha: `7be916c`
- dataset root: `/mnt/storage/research/nuscenes`
- artifact root: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v9`
- attempted invocations: `2`
- completed invocations: `2`
- memory mode: `server`
- memory embedder: `fastembed`
- planner provider: `harness:proposal_aligned_v1`
- critic provider: `openai`
- active phase: `post_run_sync`
- active checklist item: `Memory And State / Verify that the pre-run brief cites the real incumbent and active branch.`
- planner bottleneck: `geometry-bridge-gap`
- planner objective: `replace the failing DINO-on-student path with a lightweight gated bridge student and teacher-side frontier supervision`
- planner decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v9/invocation_002_20260409-181232/planner_decision.json`
- critic decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v9/invocation_002_20260409-181232/critic_decision.json`
- proposal path: `-`
- last invocation dir: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v9/invocation_002_20260409-181232`
- last selected recipe: `None`
- last NDS: `0.0`
- last mAP: `0.0`
- last publish status: `published`
- last publish message: `autoresearch: publish invocation_001_20260409-180417 (2026-04-09 18:12 UTC)`

## Notes
- started invocation `invocation_002_20260409-181232` at `2026-04-09T18:12:49.620438+00:00`
- wrote first-principles checkpoint to `artifacts/autoresearch_frontier_v9/invocation_002_20260409-181232/first_principles_checkpoint.json` before launch
- planner provider `harness:proposal_aligned_v1` selected bottleneck `geometry-bridge-gap`
- starting bounded post-run maintenance

## Pointers
- current PI brief: [docs/reports/current.md](docs/reports/current.md)
- supervisor ledger: [artifacts/autoresearch_frontier_v9/ledger.jsonl](artifacts/autoresearch_frontier_v9/ledger.jsonl)
- supervisor stop file: `artifacts/autoresearch_frontier_v9/STOP`
