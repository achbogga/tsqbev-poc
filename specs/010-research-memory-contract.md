# Spec 010: Research Memory Contract

## Goal

Maintain a rebuildable, local-first memory layer that preserves research findings beyond transient
chat context and makes the bounded loop operate from durable evidence.

## Canonical Truth

- Repo files and generated artifacts remain the canonical source of truth.
- The local memory databases are rebuildable caches, not the source of authority.
- Numeric decisions must come from exact artifact parsing before any semantic retrieval result is
  allowed to influence planning.

## Required Storage Layers

- `DuckDB` exact catalog for runs, summaries, gates, blockers, and promoted records.
- `Qdrant` local evidence index for chunked docs, ledgers, scripts, specs, and machine-written
  summaries.
- structured literature databases and technique cards under `research/knowledge/` must be indexed
  as evidence and distilled into memory facts during sync.
- `Mem0 OSS` distilled memory layer when its local services are healthy.
- If `Mem0` is unavailable, memory writes must be spooled locally and flushed later.

## Required Read/Write Flow

Every bounded research invocation must:

1. build or refresh a pre-run research brief
2. read the current incumbent, scale blocker, and repeated rabbit-hole signals from exact storage
3. consult semantic evidence and distilled memory when available
4. write post-run artifacts
5. sync new evidence into the local memory stack
6. publish an updated PI-facing report

## Required Outputs

- `artifacts/memory/sync_manifest.json`
- `artifacts/memory/brief.json`
- `docs/reports/current.md`
- append-only timestamped report logs in `docs/reports/log/`

## Failure Semantics

- Memory sync failure must never invalidate a completed training or evaluation run.
- If `Qdrant` is unavailable, exact catalog queries and lexical evidence retrieval must still work.
- If `Mem0` is unavailable, distilled facts must be spooled for later replay.

## Standing Implementation Defaults

- Default exact store: `DuckDB`
- Default evidence index: `Qdrant` with local or service mode
- Default evidence embedder: `FastEmbed` when available, otherwise deterministic hash fallback
- Default distilled-memory backend: `Mem0 OSS` with local `Qdrant` + `Ollama`
- Default reports are PI-facing, concise, and citation-heavy

## References

- Mem0 OSS: <https://github.com/mem0ai/mem0>
- Mem0 local Qdrant + Ollama cookbook:
  <https://docs.mem0.ai/cookbooks/companions/local-companion-ollama>
- Qdrant local mode and retrieval docs:
  <https://qdrant.tech/documentation/frameworks/langchain/>
- Karpathy autoresearch program:
  <https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md>
