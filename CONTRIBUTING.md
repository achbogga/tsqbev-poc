# Contributing

This repo is an active research codebase. Contributions are welcome, but they need to preserve
evidence quality, reproducibility, and contributor ergonomics for downstream forks.

## Before You Start

- prefer primary sources for papers, code, and checkpoints
- do not revert unrelated local changes
- do not make benchmark claims without linking the exact artifact
- do not commit secrets, API keys, or dataset credentials

## Local Setup

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev --extra data
```

## Required Checks

Run these before opening a PR:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest -q
```

If you changed memory or knowledge-base code, also run:

```bash
uv run tsqbev memory-health
uv run tsqbev knowledge-assets-status
```

## Minimal Research Workflow

```bash
uv run tsqbev smoke
uv run tsqbev check-data --dataset-root /path/to/dataset/root
uv run tsqbev research-report
```

Artifacts should land under `artifacts/` with exact summaries and logs.

## Research Memory And Knowledge Base

The repo uses:

- DuckDB for exact research state
- Qdrant for semantic evidence
- Mem0 for optional distilled memory
- a mirrored local research asset store

Useful commands:

```bash
uv run tsqbev memory-health
uv run tsqbev memory-backfill
uv run tsqbev knowledge-assets-sync
uv run tsqbev knowledge-assets-status
```

When adding papers or techniques:

- use official papers, official repos, and official checkpoint pages
- add or update structured JSON cards under `research/knowledge/`
- update mirrored asset manifests through the knowledge-assets tooling
- include `apply_when`, `avoid_when`, intuition, and failure modes

## Datasets And External Dependencies

This repo targets public datasets and public upstreams.

Common external dependencies:

- `nuScenes`
- `OpenLane`
- `BEVFusion`
- `OpenPCDet`
- gated foundation checkpoints such as `DINOv3`

Document any new external dependency in the relevant doc under `docs/`.

## Benchmark Claims

If you claim a metric improvement:

- link the exact artifact
- report the evaluation split
- include latency/export geometry if relevant
- say whether it is a trusted control, a pilot, or a provisional result

## Generated Files

Generated reports and memory manifests are tracked in some cases. Only include their churn when it
is intentional and relevant to the change.

## Pull Requests

Good PRs are:

- scoped
- evidence-backed
- tested
- explicit about tradeoffs

If your change is a knowledge-base or doc update, say which technique cards, mirrored assets, or
agenda items were changed and why.
