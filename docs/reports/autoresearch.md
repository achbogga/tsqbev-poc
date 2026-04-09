# Autoresearch Supervisor
_Generated: `2026-04-09T18:25:08.458507+00:00`_

## Status
- status: `running`
- branch: `main`
- repo sha: `8c510bf`
- dataset root: `/mnt/storage/research/nuscenes`
- artifact root: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v9`
- attempted invocations: `3`
- completed invocations: `3`
- memory mode: `server`
- memory embedder: `fastembed`
- planner provider: `harness:proposal_aligned_v1`
- critic provider: `openai`
- active phase: `post_run_sync`
- active checklist item: `Memory And State / Verify that the pre-run brief cites the real incumbent and active branch.`
- planner bottleneck: `geometry-bridge-gap`
- planner objective: `replace the failing DINO-on-student path with a lightweight gated bridge student and teacher-side frontier supervision`
- planner decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v9/invocation_003_20260409-182041/planner_decision.json`
- critic decision path: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v9/invocation_003_20260409-182041/critic_decision.json`
- proposal path: `-`
- last invocation dir: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier_v9/invocation_003_20260409-182041`
- last selected recipe: `None`
- last NDS: `0.0`
- last mAP: `0.0`
- last publish status: `published`
- last publish message: `autoresearch: publish invocation_002_20260409-181232 (2026-04-09 18:20 UTC)`

## Notes
- started invocation `invocation_003_20260409-182041` at `2026-04-09T18:20:58.824587+00:00`
- wrote first-principles checkpoint to `artifacts/autoresearch_frontier_v9/invocation_003_20260409-182041/first_principles_checkpoint.json` before launch
- planner provider `harness:proposal_aligned_v1` selected bottleneck `geometry-bridge-gap`
- critic rejected the planner proposal, but fallback execution is enabled; running the bounded loop with the critic policy instead of idling the GPU
- starting bounded post-run maintenance

## Pointers
- current PI brief: [docs/reports/current.md](docs/reports/current.md)
- supervisor ledger: [artifacts/autoresearch_frontier_v9/ledger.jsonl](artifacts/autoresearch_frontier_v9/ledger.jsonl)
- supervisor stop file: `artifacts/autoresearch_frontier_v9/STOP`
