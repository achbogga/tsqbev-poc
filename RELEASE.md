# Release And Fork Bootstrap

This repo is designed to be forkable by outside researchers. A good release is not just a git tag;
it is a reproducible handoff with enough exact state that another lab can bootstrap without reading
the entire internal history.

## Release Goal

A release should provide:

- one trusted local control result
- one reproduced strong upstream teacher/control
- a live research agenda
- a working memory/knowledge layer
- clean bootstrap instructions for forks

## Fork Bootstrap Checklist

```bash
git clone <repo>
cd tsqbev-poc
uv venv
source .venv/bin/activate
uv sync --extra dev --extra data
uv run pytest -q
uv run tsqbev memory-health
uv run tsqbev research-report
uv run tsqbev knowledge-assets-status
```

If datasets are available locally:

```bash
uv run tsqbev check-data --dataset-root /path/to/dataset/root
```

## Minimum Artifact Pack

At release time, make sure these are present and up to date:

- trusted local control artifact
- reproduced upstream baseline artifact
- current promoted memory manifest
- current brief/report
- knowledge asset coverage summary
- live research agenda

Current examples:

- [artifacts/research_v29_continuation_v1/research_loop/summary.json](artifacts/research_v29_continuation_v1/research_loop/summary.json)
- [artifacts/bevfusion_repro/bevfusion_bbox_summary.json](artifacts/bevfusion_repro/bevfusion_bbox_summary.json)
- [artifacts/memory/sync_manifest.json](artifacts/memory/sync_manifest.json)
- [artifacts/knowledge_assets/coverage_summary.json](artifacts/knowledge_assets/coverage_summary.json)
- [docs/research-agenda.md](docs/research-agenda.md)

## Secrets And Access

Never commit:

- API keys
- dataset credentials
- gated checkpoint tokens

Common optional env vars for operators:

- `COHERE_API_KEY`
- `OPENAI_API_KEY`
- `WANDB_API_KEY`

Gated model access should be documented, never embedded in tracked files.

## Release Verification

Before tagging or telling others to fork:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest -q
uv run tsqbev memory-health
uv run tsqbev research-report
uv run tsqbev knowledge-assets-status
```

Verify that:

- the trusted control artifact is still present
- the upstream baseline artifact is still present
- the current memory manifest points at a promoted build
- README and agenda match the current exact state

## Known External Blockers

Outside contributors may still need:

- licensed dataset access for `nuScenes` and `OpenLane`
- gated checkpoint approval for some foundation models
- local GPU support for heavier training/eval paths

Keep those blockers documented in the relevant runbook instead of hiding them in chat history.

## What A Fork Should Publish

When publishing a fork or derivative release, prefer:

- exact local-control artifact links
- exact upstream-control artifact links
- the current agenda and memory status
- a short note on which external assets are mirrored locally and which remain gated
