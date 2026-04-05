# Research Assets

The repo now treats official external research sources as a mirrored local asset store rather than
just URLs in docs.

## Layout

- local root: `/home/achbogga/projects/research_assets`
- generated manifest: [artifacts/knowledge_assets/manifest.json](../artifacts/knowledge_assets/manifest.json)
- generated coverage summary:
  [artifacts/knowledge_assets/coverage_summary.json](../artifacts/knowledge_assets/coverage_summary.json)

## Commands

```bash
uv run tsqbev knowledge-assets-sync
uv run tsqbev knowledge-assets-status
```

`knowledge-assets-sync` mirrors official source assets referenced from the structured knowledge
packs under `research/knowledge/`.

Current policy:

- official Git repos are cloned shallow into the local asset store
- papers, project pages, and checkpoints are mirrored as local files when the source allows it
- failures are recorded in the generated manifest instead of silently disappearing

This keeps the planner, critic, and future retrieval flows grounded in local, quickly retrievable
source material rather than relying only on outbound URLs.
