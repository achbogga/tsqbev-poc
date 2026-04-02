# Research Memory

`tsqbev-poc` now carries a local-first research memory stack so the bounded loop can operate from
durable evidence instead of ephemeral chat context.

## Stack

- exact catalog: `DuckDB`
- evidence index: `Qdrant`
- distilled long-term memory: `Mem0 OSS`
- local reports: `docs/reports/current.md` and `docs/reports/log/`
- local model/runtime helper services: `Qdrant` + `Ollama`

This repo intentionally keeps the canonical truth in the filesystem. The memory databases are
rebuildable from repo artifacts and docs.

## Why This Shape

- `DuckDB` is the exact source for runs, gates, blockers, and incumbents.
- `Qdrant` gives a free self-hosted evidence index with local or service-backed operation.
- `Mem0` is used as the durable memory layer for distilled findings when local services are up.
- The loop degrades gracefully: if `Mem0` or `Qdrant` are unavailable, exact catalog queries still
  work and pending memory writes are spooled.

## Commands

```bash
uv run tsqbev memory-health
uv run tsqbev memory-backfill
uv run tsqbev memory-query --query "why is scale-up blocked?"
uv run tsqbev research-brief
uv run tsqbev research-report
uv run tsqbev research-supervisor \
  --dataset-root /home/achbogga/projects/research/nuscenes \
  --artifact-dir artifacts/autoresearch \
  --teacher-cache-dir artifacts/teacher_cache/centerpoint_pointpillar_mini
```

Optional helper services:

```bash
uv run tsqbev memory-up
uv run tsqbev memory-down
```

The service helper uses [research/memory/docker-compose.yaml](../research/memory/docker-compose.yaml)
to start `Qdrant` and `Ollama`. The exact catalog and report generation do not require those
services.

To enable the full `Mem0` path instead of local exact+semantic degraded mode:

```bash
uv run tsqbev memory-up
docker compose -f research/memory/docker-compose.yaml exec ollama \
  ollama pull llama3.1:latest
docker compose -f research/memory/docker-compose.yaml exec ollama \
  ollama pull nomic-embed-text:latest
uv run tsqbev memory-health
```

If those services are unavailable, the repo still works:

- exact catalog queries continue from `DuckDB`
- lexical and semantic evidence retrieval continue locally
- distilled memory writes are spooled for later replay

## Outputs

- machine-readable sync state: [artifacts/memory/sync_manifest.json](../artifacts/memory/sync_manifest.json)
- machine-readable brief: [artifacts/memory/brief.json](../artifacts/memory/brief.json)
- PI-facing current report: [docs/reports/current.md](reports/current.md)
- continuous supervisor report: [docs/reports/autoresearch.md](reports/autoresearch.md)
- steering file consumed before runs: [docs/steering.md](steering.md)

## References

- Mem0 OSS: <https://github.com/mem0ai/mem0>
- Mem0 local companion cookbook:
  <https://docs.mem0.ai/cookbooks/companions/local-companion-ollama>
- Qdrant local mode:
  <https://qdrant.tech/documentation/frameworks/langchain/>
- Qdrant reranker guide:
  <https://qdrant.tech/documentation/fastembed/fastembed-rerankers/>
