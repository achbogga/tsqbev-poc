"""Local research memory, evidence indexing, and PI briefing.

References:
- Mem0 open-source memory layer:
  https://github.com/mem0ai/mem0
- Cohere Rerank overview:
  https://docs.cohere.com/docs/rerank-overview
- Mem0 local Ollama + Qdrant configuration:
  https://docs.mem0.ai/cookbooks/companions/local-companion-ollama
- Qdrant local mode and hybrid retrieval overview:
  https://qdrant.tech/documentation/frameworks/langchain/
- Qdrant FastEmbed rerankers:
  https://qdrant.tech/documentation/fastembed/fastembed-rerankers/
- DuckDB documentation:
  https://duckdb.org/docs/stable/
- Karpathy autoresearch program:
  https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.error import URLError
from urllib.request import urlopen

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_ROOT = REPO_ROOT / ".local" / "research-memory"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "memory"
DEFAULT_REPORTS_ROOT = REPO_ROOT / "docs" / "reports"
DEFAULT_REPORT_LOG_ROOT = DEFAULT_REPORTS_ROOT / "log"
DEFAULT_STEERING_PATH = REPO_ROOT / "docs" / "steering.md"
DEFAULT_DOCKER_COMPOSE = REPO_ROOT / "research" / "memory" / "docker-compose.yaml"
GENERATED_MEMORY_PREFIXES = (
    "docs/reports/",
    "artifacts/memory/",
    ".local/",
)
LOW_VALUE_EVIDENCE_SOURCES = {
    "src/tsqbev/research_memory.py",
    "docs/research-memory.md",
    "specs/010-research-memory-contract.md",
    "research/memory/docker-compose.yaml",
}

try:  # pragma: no cover - import availability is exercised by health checks and tests.
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]

Memory: Any = None
try:  # pragma: no cover - optional runtime dependency.
    from mem0 import Memory as _Mem0Memory
except ImportError:  # pragma: no cover
    _Mem0Memory = None
Memory = _Mem0Memory

QdrantClient: Any = None
Distance: Any = None
PointStruct: Any = None
VectorParams: Any = None
_QdrantClient: Any = None
try:  # pragma: no cover - optional runtime dependency.
    from qdrant_client import QdrantClient as _ImportedQdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    _QdrantClient = _ImportedQdrantClient
except ImportError:  # pragma: no cover
    pass
QdrantClient = cast(Any, _QdrantClient)

TextEmbedding: Any = None
try:  # pragma: no cover - optional runtime dependency.
    from fastembed import TextEmbedding as _TextEmbedding
except ImportError:  # pragma: no cover
    _TextEmbedding = None
TextEmbedding = _TextEmbedding

TextCrossEncoder: Any = None
try:  # pragma: no cover - optional runtime dependency.
    from fastembed.rerank.cross_encoder import TextCrossEncoder as _TextCrossEncoder
except ImportError:  # pragma: no cover
    _TextCrossEncoder = None
TextCrossEncoder = _TextCrossEncoder

CohereClientV2: Any = None
try:  # pragma: no cover - optional runtime dependency.
    from cohere import ClientV2 as _CohereClientV2
except ImportError:  # pragma: no cover
    _CohereClientV2 = None
CohereClientV2 = _CohereClientV2


class ResearchMemoryConfig(BaseModel):
    """Typed runtime configuration for the local research-memory stack."""

    enabled: bool = True
    memory_root: Path = Field(default_factory=lambda: DEFAULT_MEMORY_ROOT)
    artifact_root: Path = Field(default_factory=lambda: DEFAULT_ARTIFACT_ROOT)
    reports_root: Path = Field(default_factory=lambda: DEFAULT_REPORTS_ROOT)
    report_log_root: Path = Field(default_factory=lambda: DEFAULT_REPORT_LOG_ROOT)
    steering_path: Path = Field(default_factory=lambda: DEFAULT_STEERING_PATH)
    docker_compose_path: Path = Field(default_factory=lambda: DEFAULT_DOCKER_COMPOSE)
    qdrant_enabled: bool = True
    qdrant_mode: Literal["auto", "server", "local"] = "auto"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_path: Path = Field(default_factory=lambda: DEFAULT_MEMORY_ROOT / "qdrant")
    qdrant_evidence_collection: str = "tsqbev_evidence"
    qdrant_memory_collection: str = "tsqbev_memories"
    mem0_enabled: bool = True
    mem0_user_id: str = "tsqbev-research-agent"
    mem0_qdrant_collection: str = "tsqbev_mem0"
    mem0_ollama_base_url: str = "http://127.0.0.1:11434"
    mem0_ollama_llm_model: str = "llama3.1:latest"
    mem0_ollama_embed_model: str = "nomic-embed-text:latest"
    embedder_provider: Literal["auto", "fastembed", "hash"] = "auto"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_enabled: bool = False
    reranker_provider: Literal["auto", "fastembed", "cohere"] = "auto"
    reranker_model: str = "BAAI/bge-reranker-base"
    cohere_reranker_model: str = "rerank-v4.0-pro"
    cohere_api_key: str | None = None
    cohere_base_url: str | None = None
    chunk_chars: int = 1200
    chunk_overlap_chars: int = 160
    evidence_top_k: int = 8

    @property
    def catalog_path(self) -> Path:
        return self.memory_root / "catalog.duckdb"

    @property
    def mem0_spool_path(self) -> Path:
        return self.memory_root / "mem0_spool.jsonl"

    @classmethod
    def from_env(cls) -> ResearchMemoryConfig:
        def _env_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() not in {"0", "false", "off", "no"}

        return cls(
            enabled=_env_bool("TSQBEV_MEMORY_ENABLED", True),
            memory_root=Path(os.getenv("TSQBEV_MEMORY_ROOT", str(DEFAULT_MEMORY_ROOT))),
            artifact_root=Path(
                os.getenv("TSQBEV_MEMORY_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))
            ),
            reports_root=Path(os.getenv("TSQBEV_MEMORY_REPORTS_ROOT", str(DEFAULT_REPORTS_ROOT))),
            report_log_root=Path(
                os.getenv("TSQBEV_MEMORY_REPORT_LOG_ROOT", str(DEFAULT_REPORT_LOG_ROOT))
            ),
            steering_path=Path(os.getenv("TSQBEV_STEERING_PATH", str(DEFAULT_STEERING_PATH))),
            docker_compose_path=Path(
                os.getenv("TSQBEV_MEMORY_DOCKER_COMPOSE", str(DEFAULT_DOCKER_COMPOSE))
            ),
            qdrant_enabled=_env_bool("TSQBEV_QDRANT_ENABLED", True),
            qdrant_mode=os.getenv("TSQBEV_QDRANT_MODE", "auto"),  # type: ignore[arg-type]
            qdrant_url=os.getenv("TSQBEV_QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_path=Path(os.getenv("TSQBEV_QDRANT_PATH", str(DEFAULT_MEMORY_ROOT / "qdrant"))),
            mem0_enabled=_env_bool("TSQBEV_MEM0_ENABLED", True),
            mem0_user_id=os.getenv("TSQBEV_MEM0_USER_ID", "tsqbev-research-agent"),
            mem0_qdrant_collection=os.getenv("TSQBEV_MEM0_COLLECTION", "tsqbev_mem0"),
            mem0_ollama_base_url=os.getenv("TSQBEV_MEM0_OLLAMA_URL", "http://127.0.0.1:11434"),
            mem0_ollama_llm_model=os.getenv("TSQBEV_MEM0_LLM_MODEL", "llama3.1:latest"),
            mem0_ollama_embed_model=os.getenv(
                "TSQBEV_MEM0_EMBED_MODEL", "nomic-embed-text:latest"
            ),
            embedder_provider=os.getenv("TSQBEV_MEMORY_EMBEDDER", "auto"),  # type: ignore[arg-type]
            embedder_model=os.getenv(
                "TSQBEV_MEMORY_EMBEDDER_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            reranker_enabled=_env_bool("TSQBEV_MEMORY_RERANKER_ENABLED", False),
            reranker_provider=os.getenv(
                "TSQBEV_MEMORY_RERANKER_PROVIDER", "auto"
            ),  # type: ignore[arg-type]
            reranker_model=os.getenv("TSQBEV_MEMORY_RERANKER_MODEL", "BAAI/bge-reranker-base"),
            cohere_reranker_model=os.getenv(
                "TSQBEV_COHERE_RERANKER_MODEL", "rerank-v4.0-pro"
            ),
            cohere_api_key=os.getenv("TSQBEV_COHERE_API_KEY") or os.getenv("COHERE_API_KEY"),
            cohere_base_url=os.getenv("TSQBEV_COHERE_BASE_URL"),
            chunk_chars=int(os.getenv("TSQBEV_MEMORY_CHUNK_CHARS", "1200")),
            chunk_overlap_chars=int(os.getenv("TSQBEV_MEMORY_CHUNK_OVERLAP", "160")),
            evidence_top_k=int(os.getenv("TSQBEV_MEMORY_TOP_K", "8")),
        )


@dataclass(slots=True)
class ResearchEvent:
    event_id: str
    event_type: str
    source_path: str
    dataset: str | None
    architecture_family: str | None
    stage: str | None
    recipe: str | None
    parent_recipe: str | None
    status: str | None
    interim_decision: str | None
    final_decision: str | None
    git_sha: str | None
    run_id: int | None
    hypothesis: str | None
    mutation_reason: str | None
    targeted_bottleneck: str | None
    root_cause_verdict: str | None
    token_burn_score: float | None
    nds: float | None
    mean_ap: float | None
    val_total: float | None
    latency_ms: float | None
    payload: dict[str, Any]


@dataclass(slots=True)
class EvidenceChunk:
    chunk_id: str
    source_path: str
    kind: str
    title: str
    text: str
    citation: str
    payload: dict[str, Any]


@dataclass(slots=True)
class MemoryFact:
    fact_id: str
    kind: str
    claim: str
    confidence: float
    source_refs: list[str]
    dataset: str | None
    architecture_family: str | None
    bottleneck: str | None
    git_sha: str | None
    run_id: int | None
    payload: dict[str, Any]


@dataclass(slots=True)
class PIBrief:
    generated_at_utc: str
    current_state: list[str]
    delta_since_last: list[str]
    open_blockers: list[str]
    recommended_next_steps: list[str]
    evidence_refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        sections = [
            "# Research Brief",
            f"_Generated: `{self.generated_at_utc}`_",
            "",
            "## Current State",
            *[f"- {line}" for line in self.current_state],
            "",
            "## Delta Since Last Brief",
            *[f"- {line}" for line in self.delta_since_last],
            "",
            "## Open Blockers",
            *[f"- {line}" for line in self.open_blockers],
            "",
            "## Recommended Next Steps",
            *[f"- {line}" for line in self.recommended_next_steps],
            "",
            "## Evidence",
            *[f"- {line}" for line in self.evidence_refs],
            "",
        ]
        return "\n".join(sections)


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _repo_rel(path: Path, repo_root: Path = REPO_ROOT) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _repo_link(path: Path, repo_root: Path = REPO_ROOT) -> str:
    rel = _repo_rel(path, repo_root)
    return f"[{rel}]({rel})"


def _heading_chunks(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    blocks = re.split(r"(?m)^(?=#|##|###|\*\*)", stripped)
    chunks: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= chunk_chars:
            chunks.append(block)
            continue
        start = 0
        while start < len(block):
            end = min(len(block), start + chunk_chars)
            chunks.append(block[start:end].strip())
            if end >= len(block):
                break
            start = max(0, end - overlap_chars)
    return [chunk for chunk in chunks if chunk]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _hash_embed(text: str, *, dims: int = 256) -> list[float]:
    counts = Counter(_tokenize(text))
    vector = [0.0] * dims
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dims
        sign = -1.0 if digest[4] % 2 else 1.0
        vector[index] += sign * float(count)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0.0:
        vector = [value / norm for value in vector]
    return vector


def _qdrant_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


class _Embedder:
    def __init__(self, config: ResearchMemoryConfig) -> None:
        self.config = config
        self.provider = "hash"
        self.dimension = 256
        self.reranker_provider = "none"
        self.reranker_reason: str | None = None
        self._fastembed_model: Any | None = None
        if config.embedder_provider in {"auto", "fastembed"} and TextEmbedding is not None:
            try:
                self._fastembed_model = TextEmbedding(model_name=config.embedder_model)
                sample = next(iter(self._fastembed_model.embed(["tsqbev research memory"])))
                self.dimension = len(list(sample))
                self.provider = "fastembed"
            except Exception:
                self._fastembed_model = None
        self._reranker: Any | None = None
        self._cohere_client: Any | None = None
        if config.reranker_enabled:
            if config.reranker_provider in {"auto", "cohere"}:
                if CohereClientV2 is None:
                    self.reranker_reason = "cohere sdk is not installed"
                elif not config.cohere_api_key:
                    self.reranker_reason = "COHERE_API_KEY is not configured"
                else:
                    try:
                        kwargs: dict[str, Any] = {"api_key": config.cohere_api_key}
                        if config.cohere_base_url is not None:
                            kwargs["base_url"] = config.cohere_base_url
                        self._cohere_client = CohereClientV2(**kwargs)
                        self.reranker_provider = "cohere"
                    except Exception as exc:
                        self.reranker_reason = repr(exc)
                        self._cohere_client = None
            if (
                self._cohere_client is None
                and config.reranker_provider in {"auto", "fastembed"}
                and TextCrossEncoder is not None
            ):
                try:
                    self._reranker = TextCrossEncoder(model_name=config.reranker_model)
                    self.reranker_provider = "fastembed"
                    self.reranker_reason = None
                except Exception as exc:
                    self._reranker = None
                    self.reranker_reason = repr(exc)
            elif self._cohere_client is None and self.reranker_provider == "none":
                self.reranker_reason = self.reranker_reason or "no reranker backend available"

    @property
    def reranker_enabled(self) -> bool:
        return self._cohere_client is not None or self._reranker is not None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._fastembed_model is not None:
            return [list(vector) for vector in self._fastembed_model.embed(texts)]
        return [_hash_embed(text, dims=self.dimension) for text in texts]

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        text_key: str = "text",
    ) -> list[dict[str, Any]]:
        if not candidates:
            return candidates
        texts = [str(candidate[text_key]) for candidate in candidates]
        results: list[Any]
        if self._cohere_client is not None:
            try:
                response = self._cohere_client.rerank(
                    model=self.config.cohere_reranker_model,
                    query=query,
                    documents=texts,
                    top_n=len(texts),
                )
                results = list(getattr(response, "results", []))
            except Exception:
                return candidates
        elif self._reranker is not None:
            try:
                results = list(self._reranker.rerank(query, texts))
            except Exception:
                return candidates
        else:
            return candidates
        scores_by_index: dict[int, float] = {}
        for _index, result in enumerate(results):
            if isinstance(result, tuple) and len(result) >= 2:
                scores_by_index[int(result[0])] = float(result[1])
            elif hasattr(result, "relevance_score") and hasattr(result, "index"):
                scores_by_index[int(result.index)] = float(result.relevance_score)
            elif hasattr(result, "score") and hasattr(result, "index"):
                scores_by_index[int(result.index)] = float(result.score)
        reranked = []
        for index, candidate in enumerate(candidates):
            payload = dict(candidate)
            payload["rerank_score"] = scores_by_index.get(index, 0.0)
            reranked.append(payload)
        reranked.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
        return reranked


def _semantic_rank_key(item: dict[str, Any]) -> tuple[float, float, float]:
    source_path = str(item.get("source_path", ""))
    priority = _evidence_priority(source_path)
    primary_score = _safe_float(item.get("rerank_score"))
    if primary_score is None:
        primary_score = _safe_float(item.get("semantic_score")) or 0.0
    semantic_score = _safe_float(item.get("semantic_score")) or 0.0
    return (
        primary_score * priority,
        semantic_score * priority,
        priority,
    )


class ResearchCatalog:
    """Exact local analytics catalog for research memory."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        if duckdb is None:  # pragma: no cover - exercised by CI dependency install.
            raise RuntimeError("duckdb is required for the research memory catalog")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.read_only = read_only
        if read_only and path.exists():
            self._conn = duckdb.connect(str(path), read_only=True)
        else:
            self._conn = duckdb.connect(str(path))
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS research_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,
                source_path TEXT,
                dataset TEXT,
                architecture_family TEXT,
                stage TEXT,
                recipe TEXT,
                parent_recipe TEXT,
                status TEXT,
                interim_decision TEXT,
                final_decision TEXT,
                git_sha TEXT,
                run_id BIGINT,
                hypothesis TEXT,
                mutation_reason TEXT,
                targeted_bottleneck TEXT,
                root_cause_verdict TEXT,
                token_burn_score DOUBLE,
                nds DOUBLE,
                mean_ap DOUBLE,
                val_total DOUBLE,
                latency_ms DOUBLE,
                payload_json TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_chunks (
                chunk_id TEXT PRIMARY KEY,
                source_path TEXT,
                kind TEXT,
                title TEXT,
                text TEXT,
                citation TEXT,
                payload_json TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_facts (
                fact_id TEXT PRIMARY KEY,
                kind TEXT,
                claim TEXT,
                confidence DOUBLE,
                source_refs_json TEXT,
                dataset TEXT,
                architecture_family TEXT,
                bottleneck TEXT,
                git_sha TEXT,
                run_id BIGINT,
                payload_json TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_runs (
                sync_id TEXT PRIMARY KEY,
                generated_at_utc TEXT,
                repo_sha TEXT,
                evidence_count BIGINT,
                event_count BIGINT,
                fact_count BIGINT,
                payload_json TEXT
            )
            """
        )

    def close(self) -> None:
        self._conn.close()

    def upsert_events(self, events: Iterable[ResearchEvent]) -> int:
        count = 0
        for event in events:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO research_events
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.source_path,
                    event.dataset,
                    event.architecture_family,
                    event.stage,
                    event.recipe,
                    event.parent_recipe,
                    event.status,
                    event.interim_decision,
                    event.final_decision,
                    event.git_sha,
                    event.run_id,
                    event.hypothesis,
                    event.mutation_reason,
                    event.targeted_bottleneck,
                    event.root_cause_verdict,
                    event.token_burn_score,
                    event.nds,
                    event.mean_ap,
                    event.val_total,
                    event.latency_ms,
                    _json_dumps(event.payload),
                ),
            )
            count += 1
        return count

    def upsert_chunks(self, chunks: Iterable[EvidenceChunk]) -> int:
        count = 0
        for chunk in chunks:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO evidence_chunks VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.source_path,
                    chunk.kind,
                    chunk.title,
                    chunk.text,
                    chunk.citation,
                    _json_dumps(chunk.payload),
                ),
            )
            count += 1
        return count

    def upsert_facts(self, facts: Iterable[MemoryFact]) -> int:
        count = 0
        for fact in facts:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memory_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.fact_id,
                    fact.kind,
                    fact.claim,
                    fact.confidence,
                    _json_dumps(fact.source_refs),
                    fact.dataset,
                    fact.architecture_family,
                    fact.bottleneck,
                    fact.git_sha,
                    fact.run_id,
                    _json_dumps(fact.payload),
                ),
            )
            count += 1
        return count

    def record_sync(self, payload: dict[str, Any]) -> None:
        sync_id = _sha1(_json_dumps(payload))
        self._conn.execute(
            """
            INSERT OR REPLACE INTO sync_runs VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sync_id,
                payload["generated_at_utc"],
                payload.get("repo_sha"),
                payload.get("evidence_count"),
                payload.get("event_count"),
                payload.get("fact_count"),
                _json_dumps(payload),
            ),
        )

    def current_incumbent(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT recipe, nds, mean_ap, val_total, source_path, payload_json
            FROM research_events
            WHERE final_decision = 'promote' AND nds IS NOT NULL
            ORDER BY nds DESC, mean_ap DESC, val_total ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return {
            "recipe": row[0],
            "nds": row[1],
            "mean_ap": row[2],
            "val_total": row[3],
            "source_path": row[4],
            "payload": json.loads(str(row[5])),
        }

    def latest_scale_blocker(self) -> dict[str, Any] | None:
        rows = self._conn.execute(
            """
            SELECT recipe, event_type, source_path, payload_json
            FROM research_events
            WHERE event_type IN ('scale_gate', 'overfit_gate') AND status IN ('blocked', 'failed')
            """
        ).fetchall()
        best: dict[str, Any] | None = None
        best_key: tuple[float, float, float, float, float] | None = None
        for row in rows:
            payload = json.loads(str(row[3]))
            ratio = _payload_train_total_ratio(payload)
            nds = _payload_nds(payload)
            mean_ap = _payload_mean_ap(payload)
            car_ap = _payload_car_ap_4m(payload)
            ratio_ok = 1.0 if ratio is not None and ratio <= 0.4 else 0.0
            car_ok = 1.0 if car_ap is not None and car_ap > 0.0 else 0.0
            key = (
                ratio_ok,
                car_ok,
                nds if nds is not None else -1.0,
                mean_ap if mean_ap is not None else -1.0,
                car_ap if car_ap is not None else -1.0,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "recipe": row[0],
                    "event_type": row[1],
                    "source_path": row[2],
                    "payload": payload,
                }
        if best is None:
            return None
        return best

    def scale_blocker_for_recipe(self, recipe: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT recipe, event_type, source_path, payload_json
            FROM research_events
            WHERE event_type = 'scale_gate' AND status = 'blocked' AND recipe = ?
            ORDER BY nds DESC, mean_ap DESC, val_total ASC
            LIMIT 1
            """,
            (recipe,),
        ).fetchone()
        if row is None:
            return None
        return {
            "recipe": row[0],
            "event_type": row[1],
            "source_path": row[2],
            "payload": json.loads(str(row[3])),
        }

    def best_overfit_frontier(self) -> dict[str, Any] | None:
        rows = self._conn.execute(
            """
            SELECT recipe, source_path, payload_json
            FROM research_events
            WHERE event_type = 'overfit_gate' AND nds IS NOT NULL
            """
        ).fetchall()
        best: dict[str, Any] | None = None
        best_key: tuple[float, float, float, float] | None = None
        for row in rows:
            payload = json.loads(str(row[2]))
            nds = _payload_nds(payload)
            mean_ap = _payload_mean_ap(payload)
            car_ap = _payload_car_ap_4m(payload)
            car_ok = 1.0 if car_ap is not None and car_ap > 0.0 else 0.0
            key = (
                car_ok,
                nds if nds is not None else -1.0,
                mean_ap if mean_ap is not None else -1.0,
                car_ap if car_ap is not None else -1.0,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "recipe": row[0],
                    "source_path": row[1],
                    "payload": payload,
                }
        return best

    def best_ratio_passing_overfit_frontier(self) -> dict[str, Any] | None:
        rows = self._conn.execute(
            """
            SELECT recipe, source_path, payload_json
            FROM research_events
            WHERE event_type = 'overfit_gate' AND nds IS NOT NULL
            """
        ).fetchall()
        best: dict[str, Any] | None = None
        best_key: tuple[float, float, float, float] | None = None
        for row in rows:
            payload = json.loads(str(row[2]))
            ratio = _payload_train_total_ratio(payload)
            if ratio is None or ratio > 0.4:
                continue
            nds = _payload_nds(payload)
            mean_ap = _payload_mean_ap(payload)
            car_ap = _payload_car_ap_4m(payload)
            car_ok = 1.0 if car_ap is not None and car_ap > 0.0 else 0.0
            key = (
                car_ok,
                nds if nds is not None else -1.0,
                mean_ap if mean_ap is not None else -1.0,
                car_ap if car_ap is not None else -1.0,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "recipe": row[0],
                    "source_path": row[1],
                    "payload": payload,
                }
        return best

    def latest_upstream_baseline(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT recipe, nds, mean_ap, source_path, payload_json
            FROM research_events
            WHERE event_type = 'upstream_baseline' AND nds IS NOT NULL
            ORDER BY nds DESC, mean_ap DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return {
            "recipe": row[0],
            "nds": row[1],
            "mean_ap": row[2],
            "source_path": row[3],
            "payload": json.loads(str(row[4])),
        }

    def repeated_rabbit_holes(self, *, limit: int = 5) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT COALESCE(root_cause_verdict, 'unknown') AS verdict, COUNT(*) AS count
            FROM research_events
            WHERE event_type = 'research_run'
            GROUP BY verdict
            HAVING count >= 2 AND verdict <> 'unknown'
            ORDER BY count DESC, verdict ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [{"root_cause_verdict": row[0], "count": int(row[1])} for row in rows]

    def lexical_evidence(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        tokens = [token for token in _tokenize(query) if len(token) > 1]
        if not tokens:
            return []
        rows = self._conn.execute(
            "SELECT chunk_id, source_path, title, text, citation FROM evidence_chunks"
        ).fetchall()
        scored: list[dict[str, Any]] = []
        for row in rows:
            source_path = str(row[1])
            if _exclude_from_brief_evidence(source_path):
                continue
            text = str(row[3]).lower()
            score = sum(text.count(token) for token in tokens)
            if score <= 0:
                continue
            scored.append(
                {
                    "chunk_id": row[0],
                    "source_path": source_path,
                    "title": row[2],
                    "text": row[3],
                    "citation": row[4],
                    "score": float(score) * _evidence_priority(source_path),
                }
            )
        scored.sort(
            key=lambda item: (float(item["score"]), _evidence_priority(str(item["source_path"]))),
            reverse=True,
        )
        return _limit_distinct_sources(scored, limit)

    def recent_sync(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT payload_json FROM sync_runs ORDER BY generated_at_utc DESC LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return json.loads(str(row[0]))


class QdrantEvidenceIndex:
    """Optional semantic evidence index backed by local or service Qdrant."""

    def __init__(self, config: ResearchMemoryConfig) -> None:
        self.config = config
        self.enabled = False
        self.mode: str | None = None
        self.reason: str | None = None
        self._client: Any | None = None
        self._embedder = _Embedder(config)
        if not config.qdrant_enabled:
            self.reason = "disabled by configuration"
            return
        if QdrantClient is None or PointStruct is None or VectorParams is None or Distance is None:
            self.reason = "qdrant-client is not installed"
            return
        try:
            if config.qdrant_mode in {"auto", "server"} and _http_ok(config.qdrant_url):
                self._client = QdrantClient(url=config.qdrant_url)
                self.mode = "server"
            elif config.qdrant_mode in {"auto", "local"}:
                config.qdrant_path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(config.qdrant_path))
                self.mode = "local"
            else:
                self.reason = "configured Qdrant server is unavailable"
                return
            self.enabled = True
        except Exception as exc:
            self.reason = repr(exc)
            self._client = None
            self.enabled = False

    @property
    def embedder_provider(self) -> str:
        return self._embedder.provider

    def ensure_collection(self, collection_name: str) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            existing = {item.name for item in self._client.get_collections().collections}
            if collection_name in existing:
                return
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._embedder.dimension, distance=Distance.COSINE
                ),
            )
        except Exception as exc:
            self.reason = repr(exc)
            self.enabled = False

    def upsert_chunks(self, chunks: list[EvidenceChunk]) -> int:
        if not self.enabled or self._client is None or not chunks:
            return 0
        self.ensure_collection(self.config.qdrant_evidence_collection)
        if not self.enabled or self._client is None:
            return 0
        embeddings = self._embedder.embed_texts([chunk.text for chunk in chunks])
        points = [
            PointStruct(
                id=_qdrant_point_id(chunk.chunk_id),
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "title": chunk.title,
                    "text": chunk.text,
                    "citation": chunk.citation,
                    "kind": chunk.kind,
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        try:
            self._client.upsert(
                collection_name=self.config.qdrant_evidence_collection,
                points=points,
            )
            return len(points)
        except Exception as exc:
            self.reason = repr(exc)
            self.enabled = False
            return 0

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        if not self.enabled or self._client is None:
            return []
        self.ensure_collection(self.config.qdrant_evidence_collection)
        if not self.enabled or self._client is None:
            return []
        query_vector = self._embedder.embed_texts([query])[0]
        try:
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self.config.qdrant_evidence_collection,
                    query=query_vector,
                    limit=limit,
                    with_payload=True,
                )
                results = list(getattr(response, "points", []))
            else:
                results = self._client.search(
                    collection_name=self.config.qdrant_evidence_collection,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )
        except Exception as exc:
            self.reason = repr(exc)
            self.enabled = False
            return []
        payloads = []
        for result in results:
            payload = dict(result.payload or {})
            payload["semantic_score"] = float(result.score)
            payloads.append(payload)
        reranked = self._embedder.rerank(query, payloads)
        filtered = [
            payload
            for payload in reranked
            if not _exclude_from_brief_evidence(str(payload.get("source_path", "")))
        ]
        filtered.sort(key=_semantic_rank_key, reverse=True)
        return _limit_distinct_sources(filtered, limit)


class Mem0MemoryBackend:
    """Optional distilled-memory backend using Mem0 OSS."""

    def __init__(self, config: ResearchMemoryConfig) -> None:
        self.config = config
        self.enabled = False
        self.reason: str | None = None
        self._memory: Any | None = None
        if not config.mem0_enabled:
            self.reason = "disabled by configuration"
            return
        if Memory is None:
            self.reason = "mem0 is not installed"
            return
        if not _http_ok(config.qdrant_url) or not _http_ok(config.mem0_ollama_base_url):
            self.reason = "Qdrant or Ollama service is unavailable"
            return
        qdrant_host, qdrant_port = _split_host_port(config.qdrant_url, 6333)
        try:
            self._memory = Memory.from_config(
                {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": config.mem0_qdrant_collection,
                            "host": qdrant_host,
                            "port": qdrant_port,
                            "embedding_model_dims": 768,
                        },
                    },
                    "llm": {
                        "provider": "ollama",
                        "config": {
                            "model": config.mem0_ollama_llm_model,
                            "temperature": 0,
                            "max_tokens": 2000,
                            "ollama_base_url": config.mem0_ollama_base_url,
                        },
                    },
                    "embedder": {
                        "provider": "ollama",
                        "config": {
                            "model": config.mem0_ollama_embed_model,
                            "ollama_base_url": config.mem0_ollama_base_url,
                        },
                    },
                }
            )
            self.enabled = True
        except Exception as exc:
            self.reason = repr(exc)

    def add_fact(self, fact: MemoryFact) -> bool:
        if not self.enabled or self._memory is None:
            return False
        metadata = {
            "fact_id": fact.fact_id,
            "kind": fact.kind,
            "dataset": fact.dataset,
            "architecture_family": fact.architecture_family,
            "bottleneck": fact.bottleneck,
            "git_sha": fact.git_sha,
            "run_id": fact.run_id,
            "source_refs": fact.source_refs,
            **fact.payload,
        }
        try:
            self._memory.add(
                fact.claim,
                user_id=self.config.mem0_user_id,
                metadata=metadata,
                infer=False,
            )
            return True
        except Exception:
            return False

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        if not self.enabled or self._memory is None:
            return []
        try:
            result = self._memory.search(query, user_id=self.config.mem0_user_id, limit=limit)
        except Exception:
            return []
        if isinstance(result, dict):
            raw = result.get("results", [])
            if isinstance(raw, list):
                return [dict(item) for item in raw if isinstance(item, dict)]
        return []


def _http_ok(url: str) -> bool:
    try:
        with urlopen(url, timeout=1.5) as response:
            return int(getattr(response, "status", 200)) < 500
    except (URLError, ValueError, TimeoutError):
        return False


def _split_host_port(url: str, default_port: int) -> tuple[str, int]:
    cleaned = url.removeprefix("http://").removeprefix("https://")
    host, _, port = cleaned.partition(":")
    return host or "127.0.0.1", int(port or default_port)


def _artifact_files(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    candidates = [
        repo_root / "README.md",
        repo_root / "program.md",
        repo_root / "AGENTS.md",
        repo_root / "docs" / "plan.md",
        repo_root / "docs" / "steering.md",
    ]
    for pattern in (
        "docs/**/*.md",
        "docs/**/*.tex",
        "specs/**/*.md",
        "src/tsqbev/*.py",
        "research/scripts/*.py",
        "research/scripts/*.sh",
        "artifacts/**/summary.json",
        "artifacts/**/*_summary.json",
        "artifacts/**/results.jsonl",
        "artifacts/**/results.tsv",
        "artifacts/**/manifest.json",
        "artifacts/**/metrics_summary.json",
        "artifacts/**/calibration_summary.json",
    ):
        candidates.extend(repo_root.glob(pattern))
    for candidate in candidates:
        rel = _repo_rel(candidate, repo_root)
        if any(rel.startswith(prefix) for prefix in GENERATED_MEMORY_PREFIXES):
            continue
        if candidate.is_file() and candidate not in paths and candidate.stat().st_size <= 2_000_000:
            paths.append(candidate)
    return sorted(paths)


def _chunk_source_file(
    path: Path,
    config: ResearchMemoryConfig,
    *,
    repo_root: Path,
) -> list[EvidenceChunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = _repo_rel(path, repo_root)
    chunks = _heading_chunks(
        text,
        chunk_chars=config.chunk_chars,
        overlap_chars=config.chunk_overlap_chars,
    )
    evidence: list[EvidenceChunk] = []
    for index, chunk in enumerate(chunks):
        chunk_id = _sha1(f"{rel}:{index}:{chunk}")
        title = f"{rel} [{index + 1}]"
        evidence.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                source_path=rel,
                kind=path.suffix.lstrip(".") or "text",
                title=title,
                text=chunk,
                citation=_repo_link(path, repo_root),
                payload={"index": index, "lineage": rel},
            )
        )
    return evidence


def _parse_results_jsonl(path: Path, repo_root: Path) -> list[ResearchEvent]:
    events: list[ResearchEvent] = []
    rel = _repo_rel(path, repo_root)
    for index, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        config = payload.get("config", {})
        if not isinstance(config, dict):
            config = {}
        event_id = _sha1(f"{rel}:{index}:{line}")
        events.append(
            ResearchEvent(
                event_id=event_id,
                event_type="research_run",
                source_path=rel,
                dataset="nuScenes v1.0-mini",
                architecture_family=_architecture_family_from_payload(config),
                stage=_as_opt_str(payload.get("stage")),
                recipe=_as_opt_str(payload.get("recipe")),
                parent_recipe=_as_opt_str(payload.get("parent_recipe")),
                status=_as_opt_str(payload.get("status")),
                interim_decision=_as_opt_str(payload.get("interim_decision")),
                final_decision=_as_opt_str(payload.get("final_decision")),
                git_sha=_as_opt_str(payload.get("git_sha")),
                run_id=_safe_int(payload.get("run_id")),
                hypothesis=_as_opt_str(payload.get("hypothesis")),
                mutation_reason=_as_opt_str(payload.get("mutation_reason")),
                targeted_bottleneck=_as_opt_str(payload.get("targeted_bottleneck")),
                root_cause_verdict=_as_opt_str(payload.get("root_cause_verdict")),
                token_burn_score=_safe_float(payload.get("token_burn_score")),
                nds=_safe_float(_nested_get(payload, "evaluation", "nd_score")),
                mean_ap=_safe_float(_nested_get(payload, "evaluation", "mean_ap")),
                val_total=_safe_float(_nested_get(payload, "val", "total")),
                latency_ms=_safe_float(_nested_get(payload, "benchmark", "mean_ms")),
                payload=payload,
            )
        )
    return events


def _parse_summary_json(path: Path, repo_root: Path) -> list[ResearchEvent]:
    payload = json.loads(path.read_text())
    rel = _repo_rel(path, repo_root)
    events: list[ResearchEvent] = []
    event_type = "artifact_summary"
    status = _as_opt_str(payload.get("status"))
    dataset = None
    architecture = None
    recipe = _as_opt_str(payload.get("selected_recipe"))
    nds = None
    mean_ap = None
    val_total = None

    if "reference_workflow" in payload and "scale_gate_verdict" in payload:
        event_type = "research_summary"
        dataset = "nuScenes v1.0-mini"
        selected_record = payload.get("selected_record", {})
        if isinstance(selected_record, dict):
            config = selected_record.get("config", {})
            if isinstance(config, dict):
                architecture = _architecture_family_from_payload(config)
            nds = _safe_float(_nested_get(selected_record, "evaluation", "nd_score"))
            mean_ap = _safe_float(_nested_get(selected_record, "evaluation", "mean_ap"))
            val_total = _safe_float(_nested_get(selected_record, "val", "total"))
        scale_gate = payload.get("scale_gate_verdict", {})
        if isinstance(scale_gate, dict) and not bool(scale_gate.get("authorized", False)):
            events.append(
                ResearchEvent(
                    event_id=_sha1(f"{rel}:scale_gate"),
                    event_type="scale_gate",
                    source_path=rel,
                    dataset=dataset,
                    architecture_family=architecture,
                    stage=None,
                    recipe=recipe,
                    parent_recipe=None,
                    status="blocked",
                    interim_decision=None,
                    final_decision=None,
                    git_sha=None,
                    run_id=None,
                    hypothesis=None,
                    mutation_reason=None,
                    targeted_bottleneck="scale gate",
                    root_cause_verdict=_as_opt_str(scale_gate.get("reason")),
                    token_burn_score=None,
                    nds=nds,
                    mean_ap=mean_ap,
                    val_total=val_total,
                    latency_ms=None,
                    payload=payload,
                )
            )
    elif "subset_size" in payload and "gate_verdict" in payload:
        event_type = "overfit_gate"
        dataset = "nuScenes subset"
        recipe = _as_opt_str(payload.get("recipe"))
        if recipe is None:
            try:
                recipe = path.parent.parent.name
            except IndexError:
                recipe = None
        evaluation = payload.get("evaluation", {})
        if isinstance(evaluation, dict):
            nds = _safe_float(evaluation.get("nd_score"))
            mean_ap = _safe_float(evaluation.get("mean_ap"))
        train_payload = payload.get("train", {})
        if isinstance(train_payload, dict):
            val_total = _safe_float(train_payload.get("final_val_total"))
        gate_verdict = payload.get("gate_verdict", {})
        status = "passed" if bool(gate_verdict.get("passed", False)) else "failed"
    elif _extract_upstream_detection_metrics(payload) is not None:
        event_type = "upstream_baseline"
        dataset = "nuScenes val"
        recipe = _derive_upstream_recipe(path, payload)
        architecture = _derive_upstream_architecture(path, payload)
        metrics = _extract_upstream_detection_metrics(payload) or {}
        nds = _safe_float(metrics.get("NDS"))
        mean_ap = _safe_float(metrics.get("mAP"))
        status = "completed"
    elif "coverage" in payload:
        event_type = "teacher_cache_audit"
        status = "completed"

    events.append(
        ResearchEvent(
            event_id=_sha1(f"{rel}:{event_type}"),
            event_type=event_type,
            source_path=rel,
            dataset=dataset,
            architecture_family=architecture,
            stage=None,
            recipe=recipe,
            parent_recipe=None,
            status=status,
            interim_decision=None,
            final_decision=None,
            git_sha=None,
            run_id=None,
            hypothesis=None,
            mutation_reason=None,
            targeted_bottleneck=None,
            root_cause_verdict=_as_opt_str(payload.get("reason")),
            token_burn_score=None,
            nds=nds,
            mean_ap=mean_ap,
            val_total=val_total,
            latency_ms=None,
            payload=payload,
        )
    )
    return events


def _collect_events(repo_root: Path) -> list[ResearchEvent]:
    events: list[ResearchEvent] = []
    for path in _artifact_files(repo_root):
        if path.name == "results.jsonl":
            events.extend(_parse_results_jsonl(path, repo_root))
        elif path.name == "summary.json" or path.name.endswith("_summary.json"):
            events.extend(_parse_summary_json(path, repo_root))
    return events


def _collect_chunks(repo_root: Path, config: ResearchMemoryConfig) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for path in _artifact_files(repo_root):
        if path.suffix.lower() not in {".md", ".py", ".sh", ".tex", ".json", ".jsonl", ".tsv"}:
            continue
        if _repo_rel(path, repo_root) in LOW_VALUE_EVIDENCE_SOURCES:
            continue
        chunks.extend(_chunk_source_file(path, config, repo_root=repo_root))
    return chunks


def _derive_memory_facts(events: list[ResearchEvent], repo_root: Path) -> list[MemoryFact]:
    by_type: dict[str, list[ResearchEvent]] = defaultdict(list)
    for event in events:
        by_type[event.event_type].append(event)

    facts: list[MemoryFact] = []
    promoted = [
        event for event in by_type.get("research_run", []) if event.final_decision == "promote"
    ]
    promoted.sort(key=lambda item: (item.nds or -1.0, item.mean_ap or -1.0), reverse=True)
    if promoted:
        event = promoted[0]
        claim = (
            f"Current promoted local mini incumbent is `{event.recipe}` with "
            f"NDS `{event.nds:.4f}` and mAP `{event.mean_ap:.4f}`."
            if event.nds is not None and event.mean_ap is not None
            else f"Current promoted local mini incumbent is `{event.recipe}`."
        )
        facts.append(
            MemoryFact(
                fact_id=_sha1(f"incumbent:{claim}"),
                kind="incumbent",
                claim=claim,
                confidence=0.95,
                source_refs=[event.source_path],
                dataset=event.dataset,
                architecture_family=event.architecture_family,
                bottleneck=event.root_cause_verdict,
                git_sha=event.git_sha,
                run_id=event.run_id,
                payload={"recipe": event.recipe},
            )
        )

    overfit_events = [event for event in by_type.get("overfit_gate", []) if event.nds is not None]
    overfit_events.sort(key=lambda item: (item.nds or -1.0, item.mean_ap or -1.0), reverse=True)
    if overfit_events:
        event = overfit_events[0]
        ratio = _payload_train_total_ratio(event.payload)
        car_ap = _payload_car_ap_4m(event.payload)
        claim = (
            f"Best local overfit frontier is `{event.recipe}` with "
            f"NDS `{event.nds:.4f}`, mAP `{event.mean_ap:.4f}`, "
            f"train_total_ratio `{ratio:.4f}`, and car AP@4m `{car_ap:.4f}`."
            if event.mean_ap is not None and ratio is not None and car_ap is not None
            else f"Best local overfit frontier is `{event.recipe}`."
        )
        facts.append(
            MemoryFact(
                fact_id=_sha1(f"overfit:{claim}"),
                kind="overfit_frontier",
                claim=claim,
                confidence=0.97,
                source_refs=[event.source_path],
                dataset=event.dataset,
                architecture_family=event.architecture_family,
                bottleneck=event.root_cause_verdict,
                git_sha=event.git_sha,
                run_id=event.run_id,
                payload={"recipe": event.recipe},
            )
        )

    ratio_events = [
        event
        for event in overfit_events
        if (_payload_train_total_ratio(event.payload) or 1.0) <= 0.4
    ]
    ratio_events.sort(key=lambda item: (item.nds or -1.0, item.mean_ap or -1.0), reverse=True)
    if ratio_events:
        event = ratio_events[0]
        ratio = _payload_train_total_ratio(event.payload)
        claim = (
            f"Best ratio-passing overfit candidate is `{event.recipe}` with "
            f"NDS `{event.nds:.4f}`, mAP `{event.mean_ap:.4f}`, and "
            f"train_total_ratio `{ratio:.4f}`."
            if event.mean_ap is not None and ratio is not None
            else f"Best ratio-passing overfit candidate is `{event.recipe}`."
        )
        facts.append(
            MemoryFact(
                fact_id=_sha1(f"overfit-ratio:{claim}"),
                kind="overfit_ratio_frontier",
                claim=claim,
                confidence=0.97,
                source_refs=[event.source_path],
                dataset=event.dataset,
                architecture_family=event.architecture_family,
                bottleneck=event.root_cause_verdict,
                git_sha=event.git_sha,
                run_id=event.run_id,
                payload={"recipe": event.recipe},
            )
        )

    upstream = [event for event in by_type.get("upstream_baseline", []) if event.nds is not None]
    upstream.sort(key=lambda item: (item.nds or -1.0, item.mean_ap or -1.0), reverse=True)
    if upstream:
        event = upstream[0]
        claim = (
            f"Strongest reproduced upstream baseline is `{event.recipe}` at "
            f"NDS `{event.nds:.4f}` and mAP `{event.mean_ap:.4f}`."
        )
        facts.append(
            MemoryFact(
                fact_id=_sha1(f"upstream:{claim}"),
                kind="upstream_baseline",
                claim=claim,
                confidence=0.99,
                source_refs=[event.source_path],
                dataset=event.dataset,
                architecture_family=event.architecture_family,
                bottleneck=None,
                git_sha=None,
                run_id=None,
                payload={"recipe": event.recipe},
            )
        )

    scale_events = [event for event in by_type.get("scale_gate", []) if event.status == "blocked"]
    if scale_events:
        event = scale_events[-1]
        payload = (
            event.payload.get("scale_gate_verdict", {})
            if isinstance(event.payload, dict)
            else {}
        )
        reason = payload.get("reason") if isinstance(payload, dict) else None
        claim = "Scale-up is still blocked."
        if isinstance(reason, str) and reason:
            claim = f"Scale-up is still blocked because {reason}."
        facts.append(
            MemoryFact(
                fact_id=_sha1(f"scale:{claim}"),
                kind="scale_gate",
                claim=claim,
                confidence=0.98,
                source_refs=[event.source_path],
                dataset=event.dataset,
                architecture_family=event.architecture_family,
                bottleneck="scale gate",
                git_sha=None,
                run_id=None,
                payload={"reason": reason},
            )
        )

    root_causes = Counter(
        event.root_cause_verdict
        for event in by_type.get("research_run", [])
        if event.root_cause_verdict is not None
    )
    for verdict, count in root_causes.items():
        if count < 2:
            continue
        claim = f"Repeated local bottleneck `{verdict}` has appeared `{count}` times."
        facts.append(
            MemoryFact(
                fact_id=_sha1(f"verdict:{verdict}:{count}"),
                kind="repeated_bottleneck",
                claim=claim,
                confidence=0.75,
                source_refs=sorted(
                    {
                        event.source_path
                        for event in by_type["research_run"]
                        if event.root_cause_verdict == verdict
                    }
                )[:3],
                dataset="nuScenes v1.0-mini",
                architecture_family=None,
                bottleneck=verdict,
                git_sha=None,
                run_id=None,
                payload={"count": count},
            )
        )
    return facts


def _as_opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_upstream_detection_metrics(payload: dict[str, Any]) -> dict[str, Any] | None:
    if "mAP" in payload and "NDS" in payload:
        return {"mAP": payload.get("mAP"), "NDS": payload.get("NDS")}
    headline = payload.get("headline_metrics")
    if isinstance(headline, dict) and "mAP" in headline and "NDS" in headline:
        return {"mAP": headline.get("mAP"), "NDS": headline.get("NDS")}
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and "object/map" in metrics and "object/nds" in metrics:
        return {"mAP": metrics.get("object/map"), "NDS": metrics.get("object/nds")}
    return None


def _derive_upstream_recipe(path: Path, payload: dict[str, Any]) -> str:
    config_rel = _as_opt_str(payload.get("config_rel"))
    if config_rel is not None and "bevfusion" in path.stem:
        return f"bevfusion:{Path(config_rel).stem}"
    return path.stem.replace("_summary", "")


def _derive_upstream_architecture(path: Path, payload: dict[str, Any]) -> str | None:
    joined = " ".join(
        item
        for item in (
            _as_opt_str(payload.get("config_rel")),
            _as_opt_str(payload.get("upstream_repo_root")),
            str(path),
        )
        if item is not None
    ).lower()
    if "bevfusion" in joined:
        return "bevfusion"
    if "openpcdet" in joined or "centerpoint" in joined or "pointpillar" in joined:
        return "openpcdet"
    if "bevdet" in joined or "bevdepth" in joined:
        return "bevdet"
    if "maptr" in joined:
        return "maptr"
    if "persformer" in joined:
        return "persformer"
    return None


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _architecture_family_from_payload(config: dict[str, Any]) -> str | None:
    image_backbone = _as_opt_str(config.get("image_backbone"))
    teacher_seed_mode = _as_opt_str(config.get("teacher_seed_mode"))
    views = _safe_int(config.get("views"))
    if image_backbone is None:
        return None
    parts = [f"v{views}" if views is not None else "v?", image_backbone]
    if teacher_seed_mode not in {None, "off"}:
        parts.append(str(teacher_seed_mode))
    return "-".join(parts)


def _spool_pending_facts(config: ResearchMemoryConfig, facts: list[MemoryFact]) -> None:
    if not facts:
        return
    config.memory_root.mkdir(parents=True, exist_ok=True)
    existing = set()
    if config.mem0_spool_path.exists():
        for line in config.mem0_spool_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            fact_id = payload.get("fact_id")
            if isinstance(fact_id, str):
                existing.add(fact_id)
    with config.mem0_spool_path.open("a", encoding="utf-8") as handle:
        for fact in facts:
            if fact.fact_id in existing:
                continue
            handle.write(_json_dumps(asdict(fact)))
            handle.write("\n")


def _flush_mem0_spool(config: ResearchMemoryConfig, backend: Mem0MemoryBackend) -> dict[str, int]:
    if not config.mem0_spool_path.exists():
        return {"attempted": 0, "flushed": 0, "remaining": 0}
    attempted = 0
    flushed = 0
    remaining: list[str] = []
    for line in config.mem0_spool_path.read_text().splitlines():
        if not line.strip():
            continue
        attempted += 1
        payload = json.loads(line)
        fact = MemoryFact(
            fact_id=str(payload["fact_id"]),
            kind=str(payload["kind"]),
            claim=str(payload["claim"]),
            confidence=float(payload["confidence"]),
            source_refs=list(payload["source_refs"]),
            dataset=_as_opt_str(payload.get("dataset")),
            architecture_family=_as_opt_str(payload.get("architecture_family")),
            bottleneck=_as_opt_str(payload.get("bottleneck")),
            git_sha=_as_opt_str(payload.get("git_sha")),
            run_id=_safe_int(payload.get("run_id")),
            payload=dict(payload.get("payload", {})),
        )
        if backend.add_fact(fact):
            flushed += 1
            continue
        remaining.append(line)
    if remaining:
        config.mem0_spool_path.write_text("\n".join(remaining) + "\n")
    else:
        config.mem0_spool_path.unlink(missing_ok=True)
    return {"attempted": attempted, "flushed": flushed, "remaining": len(remaining)}


def _exclude_from_brief_evidence(source_path: str) -> bool:
    normalized = source_path.strip()
    if not normalized:
        return False
    if any(normalized.startswith(prefix) for prefix in GENERATED_MEMORY_PREFIXES):
        return True
    return normalized in LOW_VALUE_EVIDENCE_SOURCES


def _evidence_priority(source_path: str) -> float:
    normalized = source_path.strip()
    if normalized.startswith("artifacts/") and normalized.endswith((".json", ".jsonl", ".tsv")):
        return 4.0
    if normalized in {"docs/plan.md", "docs/upstream-baselines.md", "program.md", "README.md"}:
        return 3.5
    if normalized.startswith("specs/"):
        return 3.0
    if normalized.startswith("docs/"):
        return 2.5
    if normalized.startswith("research/scripts/"):
        return 1.5
    if normalized.startswith("src/"):
        return 1.0
    return 1.0


def _limit_distinct_sources(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for item in items:
        source_path = str(item.get("source_path", "")).strip()
        if source_path in seen_sources:
            continue
        seen_sources.add(source_path)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def sync_research_memory(
    repo_root: Path = REPO_ROOT,
    *,
    config: ResearchMemoryConfig | None = None,
) -> dict[str, Any]:
    """Rebuild the local research-memory state from canonical repo artifacts."""

    cfg = config or ResearchMemoryConfig.from_env()
    if not cfg.enabled:
        return {
            "status": "disabled",
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            "reason": "TSQBEV_MEMORY_ENABLED is disabled",
        }
    cfg.memory_root.mkdir(parents=True, exist_ok=True)
    cfg.artifact_root.mkdir(parents=True, exist_ok=True)
    catalog = _open_catalog_with_retry(cfg.catalog_path, read_only=False)
    try:
        events = _collect_events(repo_root)
        chunks = _collect_chunks(repo_root, cfg)
        facts = _derive_memory_facts(events, repo_root)
        event_count = catalog.upsert_events(events)
        chunk_count = catalog.upsert_chunks(chunks)
        fact_count = catalog.upsert_facts(facts)

        index = QdrantEvidenceIndex(cfg)
        indexed_chunks = index.upsert_chunks(chunks)

        mem0_backend = Mem0MemoryBackend(cfg)
        synced_now = 0
        if mem0_backend.enabled:
            for fact in facts:
                if mem0_backend.add_fact(fact):
                    synced_now += 1
            spool_result = _flush_mem0_spool(cfg, mem0_backend)
        else:
            _spool_pending_facts(cfg, facts)
            spool_result = {"attempted": 0, "flushed": 0, "remaining": len(facts)}

        summary = {
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            "repo_sha": current_git_sha(repo_root),
            "repo_root": str(repo_root),
            "catalog_path": str(cfg.catalog_path),
            "event_count": event_count,
            "evidence_count": chunk_count,
            "fact_count": fact_count,
            "qdrant": {
                "enabled": index.enabled,
                "mode": index.mode,
                "reason": index.reason,
                "indexed_chunks": indexed_chunks,
                "embedder_provider": index.embedder_provider,
            },
            "mem0": {
                "enabled": mem0_backend.enabled,
                "reason": mem0_backend.reason,
                "synced_now": synced_now,
                "spool": spool_result,
            },
        }
        catalog.record_sync(summary)
        (cfg.artifact_root / "sync_manifest.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return summary
    finally:
        catalog.close()


def build_research_brief(
    repo_root: Path = REPO_ROOT,
    *,
    config: ResearchMemoryConfig | None = None,
    persist_log: bool = False,
) -> PIBrief:
    """Build the current PI-facing research brief from exact and semantic memory."""

    cfg = config or ResearchMemoryConfig.from_env()
    if not cfg.enabled:
        brief = PIBrief(
            generated_at_utc=datetime.now(tz=UTC).isoformat(),
            current_state=["Research memory is disabled for this environment."],
            delta_since_last=["No brief delta because the memory layer is disabled."],
            open_blockers=[
                "Enable TSQBEV_MEMORY_ENABLED to hydrate the exact and semantic brief."
            ],
            recommended_next_steps=[
                "Run `tsqbev memory-backfill` after enabling the memory layer."
            ],
            evidence_refs=[],
        )
        return brief
    catalog = _open_catalog_with_retry(cfg.catalog_path, read_only=True)
    try:
        incumbent = catalog.current_incumbent()
        overfit_frontier = catalog.best_overfit_frontier()
        ratio_overfit_frontier = catalog.best_ratio_passing_overfit_frontier()
        scale_blocker = None
        if incumbent is not None:
            scale_blocker = catalog.scale_blocker_for_recipe(str(incumbent["recipe"]))
        if scale_blocker is None:
            scale_blocker = catalog.latest_scale_blocker()
        upstream = catalog.latest_upstream_baseline()
        repeated = catalog.repeated_rabbit_holes()
        lexical = catalog.lexical_evidence(
            "current incumbent scale blocker bevfusion baseline", limit=cfg.evidence_top_k
        )
        index = QdrantEvidenceIndex(cfg)
        semantic = index.search(
            "current incumbent scale blocker BEVFusion baseline and next steps",
            limit=cfg.evidence_top_k,
        )
        mem0_backend = Mem0MemoryBackend(cfg)
        memories = mem0_backend.search(
            "current incumbent, scale blocker, and next best steps",
            limit=min(4, cfg.evidence_top_k),
        )

        current_state: list[str] = []
        if incumbent is not None:
            current_state.append(
                "Local incumbent: "
                f"`{incumbent['recipe']}` with NDS `{incumbent['nds']:.4f}`, "
                f"mAP `{incumbent['mean_ap']:.4f}`, "
                f"val total `{incumbent['val_total']:.4f}` "
                f"from {_repo_link(repo_root / incumbent['source_path'], repo_root)}."
            )
        else:
            current_state.append("No promoted local incumbent is indexed yet.")
        if overfit_frontier is not None:
            frontier_payload = overfit_frontier["payload"]
            frontier_nds = _payload_nds(frontier_payload)
            frontier_map = _payload_mean_ap(frontier_payload)
            frontier_ratio = _payload_train_total_ratio(frontier_payload)
            frontier_car = _payload_car_ap_4m(frontier_payload)
            current_state.append(
                "Best local overfit frontier: "
                f"`{overfit_frontier['recipe']}` with "
                f"NDS `{frontier_nds:.4f}`, mAP `{frontier_map:.4f}`, "
                f"train_total_ratio `{frontier_ratio:.4f}`, car AP@4m `{frontier_car:.4f}` "
                f"from {_repo_link(repo_root / overfit_frontier['source_path'], repo_root)}."
            )
        if ratio_overfit_frontier is not None:
            ratio_payload = ratio_overfit_frontier["payload"]
            ratio_nds = _payload_nds(ratio_payload)
            ratio_map = _payload_mean_ap(ratio_payload)
            ratio_ratio = _payload_train_total_ratio(ratio_payload)
            current_state.append(
                "Best ratio-passing overfit candidate: "
                f"`{ratio_overfit_frontier['recipe']}` with "
                f"NDS `{ratio_nds:.4f}`, mAP `{ratio_map:.4f}`, "
                f"train_total_ratio `{ratio_ratio:.4f}` "
                f"from {_repo_link(repo_root / ratio_overfit_frontier['source_path'], repo_root)}."
            )
        if upstream is not None:
            current_state.append(
                "Best reproduced upstream baseline: "
                f"`{upstream['recipe']}` with NDS `{upstream['nds']:.4f}` and "
                f"mAP `{upstream['mean_ap']:.4f}` from "
                f"{_repo_link(repo_root / upstream['source_path'], repo_root)}."
            )
        if cfg.steering_path.exists():
            current_state.append(
                f"Active steering file present at {_repo_link(cfg.steering_path, repo_root)}."
            )

        delta_since_last = [
            "Research memory is now rebuildable from canonical repo artifacts via DuckDB "
            "exact catalog, Qdrant evidence index, and optional Mem0 sync."
        ]
        if incumbent is not None and upstream is not None:
            delta_since_last.append(
                "The strongest reproduced upstream ceiling is still materially ahead of the "
                f"local incumbent by NDS `{upstream['nds'] - incumbent['nds']:.4f}`."
            )
        if ratio_overfit_frontier is not None:
            ratio_payload = ratio_overfit_frontier["payload"]
            ratio_nds = _payload_nds(ratio_payload)
            if ratio_nds is not None:
                nds_gap = 0.10 - ratio_nds
                if nds_gap > 0:
                    delta_since_last.append(
                        "The best ratio-passing overfit candidate is now only "
                        f"`{nds_gap:.4f}` NDS shy of the overfit quality gate."
                    )
                else:
                    delta_since_last.append(
                        "The best ratio-passing overfit candidate already clears the NDS "
                        "threshold, so any remaining gate miss is coming from another criterion."
                    )
        if memories:
            delta_since_last.append(
                "Distilled memory backend returned "
                f"`{len(memories)}` relevant memory items for the current brief."
            )

        open_blockers: list[str] = []
        ratio_overfit_payload = (
            ratio_overfit_frontier["payload"] if ratio_overfit_frontier is not None else None
        )
        ratio_overfit_passed = False
        if isinstance(ratio_overfit_payload, dict):
            gate_verdict = ratio_overfit_payload.get("gate_verdict")
            if isinstance(gate_verdict, dict):
                ratio_overfit_passed = bool(gate_verdict.get("passed", False))
        incumbent_nds = incumbent["nds"] if incumbent is not None else None
        frontier_nds = (
            _payload_nds(ratio_overfit_payload) if isinstance(ratio_overfit_payload, dict) else None
        )
        should_override_stale_blocker = (
            ratio_overfit_passed
            and frontier_nds is not None
            and (
                incumbent_nds is None
                or frontier_nds > incumbent_nds + 0.02
            )
        )
        if should_override_stale_blocker and ratio_overfit_frontier is not None:
            open_blockers.append(
                "Scale-up blocker: the 32-sample overfit gate is now passed, but the promoted "
                "mini-val incumbent has not been refreshed against that frontier yet; rerun a "
                "bounded `mini_train -> mini_val` experiment from "
                f"`{ratio_overfit_frontier['recipe']}` before spending more compute "
                f"({_repo_link(repo_root / ratio_overfit_frontier['source_path'], repo_root)})."
            )
        elif scale_blocker is not None:
            payload = scale_blocker["payload"]
            reason = _extract_blocker_reason(payload)
            open_blockers.append(
                f"Scale-up blocker: {reason} "
                f"({_repo_link(repo_root / scale_blocker['source_path'], repo_root)})."
            )
        for item in repeated[:3]:
            open_blockers.append(
                "Repeated rabbit-hole signal: "
                f"`{item['root_cause_verdict']}` appeared `{item['count']}` times."
            )
        if not open_blockers:
            open_blockers.append("No structured blockers are indexed yet.")

        recommended_next_steps: list[str] = []
        if should_override_stale_blocker and ratio_overfit_frontier is not None:
            recommended_next_steps.extend(
                [
                    "Promote the passed overfit frontier into a bounded `mini_train -> mini_val` "
                    "run before any more subset-only ablations.",
                    "Keep the quality-aware teacher-anchor recipe fixed and use the next run to "
                    "measure mini-val generalization and repeatability.",
                ]
            )
        elif scale_blocker is not None:
            payload = scale_blocker["payload"]
            next_steps = payload.get("recommended_next_steps")
            if isinstance(next_steps, list):
                recommended_next_steps.extend(str(step) for step in next_steps[:3])
        if not recommended_next_steps:
            recommended_next_steps.extend(
                [
                    "Read the latest scale blocker and incumbent summaries before launching "
                    "the next bounded loop.",
                    "Backfill memory after every completed run so the next brief has exact "
                    "and semantic context.",
                    "Prefer the dense-BEV reset line over legacy sparse-query work unless "
                    "new local evidence reverses that choice.",
                ]
            )

        evidence_refs = []
        if incumbent is not None:
            evidence_refs.append(
                "Exact incumbent evidence: "
                f"{_repo_link(repo_root / incumbent['source_path'], repo_root)}."
            )
        if overfit_frontier is not None:
            evidence_refs.append(
                "Exact overfit frontier evidence: "
                f"{_repo_link(repo_root / overfit_frontier['source_path'], repo_root)}."
            )
        if ratio_overfit_frontier is not None:
            evidence_refs.append(
                "Exact ratio-passing overfit evidence: "
                f"{_repo_link(repo_root / ratio_overfit_frontier['source_path'], repo_root)}."
            )
        if upstream is not None:
            evidence_refs.append(
                "Exact upstream evidence: "
                f"{_repo_link(repo_root / upstream['source_path'], repo_root)}."
            )
        if should_override_stale_blocker and ratio_overfit_frontier is not None:
            evidence_refs.append(
                "Current blocker evidence: "
                f"{_repo_link(repo_root / ratio_overfit_frontier['source_path'], repo_root)}."
            )
        elif scale_blocker is not None:
            evidence_refs.append(
                "Exact blocker evidence: "
                f"{_repo_link(repo_root / scale_blocker['source_path'], repo_root)}."
            )
        for item in lexical[:4]:
            evidence_refs.append(f"Lexical evidence: `{item['title']}` via {item['citation']}.")
        for item in semantic[:4]:
            citation = _semantic_citation(item, repo_root)
            semantic_title = item.get("title", item.get("source_path", "unknown"))
            evidence_refs.append(
                f"Semantic evidence: `{semantic_title}` via {citation}."
            )
        for item in memories[:3]:
            memory_text = (
                _as_opt_str(item.get("memory")) or _as_opt_str(item.get("text")) or "memory hit"
            )
            evidence_refs.append(f"Mem0 memory: {memory_text}.")
        if not evidence_refs:
            evidence_refs.append("No evidence hits were available beyond the exact catalog.")

        brief = PIBrief(
            generated_at_utc=datetime.now(tz=UTC).isoformat(),
            current_state=current_state,
            delta_since_last=delta_since_last,
            open_blockers=open_blockers,
            recommended_next_steps=recommended_next_steps[:5],
            evidence_refs=evidence_refs[:8],
        )

        cfg.reports_root.mkdir(parents=True, exist_ok=True)
        cfg.report_log_root.mkdir(parents=True, exist_ok=True)
        cfg.artifact_root.mkdir(parents=True, exist_ok=True)
        (cfg.artifact_root / "brief.json").write_text(json.dumps(brief.to_dict(), indent=2))
        current_path = cfg.reports_root / "current.md"
        current_path.write_text(brief.to_markdown())
        if persist_log:
            stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
            (cfg.report_log_root / f"{stamp}.md").write_text(brief.to_markdown())
        return brief
    finally:
        catalog.close()


def query_research_memory(
    query: str,
    *,
    repo_root: Path = REPO_ROOT,
    config: ResearchMemoryConfig | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Query exact, lexical, semantic, and Mem0 memory layers together."""

    cfg = config or ResearchMemoryConfig.from_env()
    if not cfg.enabled:
        return {
            "query": query,
            "exact_facts": [],
            "lexical_evidence": [],
            "semantic_evidence": [],
            "mem0_results": [],
            "status": "disabled",
        }
    catalog = _open_catalog_with_retry(cfg.catalog_path, read_only=True)
    try:
        lexical = catalog.lexical_evidence(query, limit=limit)
        fact_rows = catalog._conn.execute(
            "SELECT kind, claim, source_refs_json, confidence FROM memory_facts"
        ).fetchall()
        fact_tokens = [token for token in _tokenize(query) if len(token) > 1]
        facts_scored: list[dict[str, Any]] = []
        for row in fact_rows:
            claim = str(row[1])
            claim_lower = claim.lower()
            score = sum(claim_lower.count(token) for token in fact_tokens)
            if score <= 0:
                continue
            facts_scored.append(
                {
                    "kind": row[0],
                    "claim": claim,
                    "source_refs": json.loads(str(row[2])),
                    "confidence": row[3],
                    "score": float(score),
                }
            )
        facts_scored.sort(
            key=lambda item: (float(item["score"]), float(item["confidence"])),
            reverse=True,
        )
        facts = facts_scored[:limit]
        semantic = QdrantEvidenceIndex(cfg).search(query, limit=limit)
        mem0_results = Mem0MemoryBackend(cfg).search(query, limit=limit)
        return {
            "query": query,
            "exact_facts": facts,
            "lexical_evidence": lexical,
            "semantic_evidence": semantic,
            "mem0_results": mem0_results,
        }
    finally:
        catalog.close()


def check_research_memory_health(
    repo_root: Path = REPO_ROOT,
    *,
    config: ResearchMemoryConfig | None = None,
) -> dict[str, Any]:
    """Return health status for the local research-memory stack."""

    cfg = config or ResearchMemoryConfig.from_env()
    if not cfg.enabled:
        return {
            "repo_root": str(repo_root),
            "status": "disabled",
            "reason": "TSQBEV_MEMORY_ENABLED is disabled",
        }
    qdrant = QdrantEvidenceIndex(cfg)
    mem0_backend = Mem0MemoryBackend(cfg)
    return {
        "repo_root": str(repo_root),
        "hostname": socket.gethostname(),
        "catalog": {
            "installed": duckdb is not None,
            "path": str(cfg.catalog_path),
            "exists": cfg.catalog_path.exists(),
        },
        "qdrant": {
            "enabled": qdrant.enabled,
            "mode": qdrant.mode,
            "reason": qdrant.reason,
            "url": cfg.qdrant_url,
            "path": str(cfg.qdrant_path),
            "embedder_provider": qdrant.embedder_provider,
        },
        "reranker": {
            "enabled": qdrant._embedder.reranker_enabled,
            "provider": qdrant._embedder.reranker_provider,
            "reason": qdrant._embedder.reranker_reason,
            "configured_provider": cfg.reranker_provider,
            "configured_model": cfg.reranker_model,
            "configured_cohere_model": cfg.cohere_reranker_model,
        },
        "mem0": {
            "enabled": mem0_backend.enabled,
            "reason": mem0_backend.reason,
            "spool_path": str(cfg.mem0_spool_path),
            "spool_exists": cfg.mem0_spool_path.exists(),
        },
        "ollama": {
            "url": cfg.mem0_ollama_base_url,
            "healthy": _http_ok(cfg.mem0_ollama_base_url),
        },
        "docker_compose": {
            "path": str(cfg.docker_compose_path),
            "exists": cfg.docker_compose_path.exists(),
        },
        "reports": {
            "current": str(cfg.reports_root / "current.md"),
            "exists": (cfg.reports_root / "current.md").exists(),
        },
    }


def _open_catalog_with_retry(
    path: Path,
    *,
    read_only: bool,
    attempts: int = 180,
    delay_s: float = 0.25,
) -> ResearchCatalog:
    for attempt in range(attempts + 1):
        try:
            return ResearchCatalog(path, read_only=read_only)
        except Exception as exc:
            if "Could not set lock on file" not in str(exc) or attempt >= attempts:
                raise
            time.sleep(delay_s)
    raise RuntimeError("unreachable")


def manage_research_memory_services(
    action: Literal["up", "down"],
    *,
    config: ResearchMemoryConfig | None = None,
) -> dict[str, Any]:
    """Start or stop the local Qdrant/Ollama helper services."""

    cfg = config or ResearchMemoryConfig.from_env()
    compose_path = cfg.docker_compose_path
    if not compose_path.exists():
        raise RuntimeError(f"memory docker-compose file not found: {compose_path}")
    cmd = ["docker", "compose", "-f", str(compose_path), action]
    if action == "up":
        cmd.append("-d")
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return {
        "action": action,
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def current_git_sha(repo_root: Path = REPO_ROOT) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _extract_blocker_reason(payload: dict[str, Any]) -> str:
    scale_gate = payload.get("scale_gate_verdict")
    if isinstance(scale_gate, dict):
        reason = _as_opt_str(scale_gate.get("reason"))
        if reason is not None:
            return reason
    gate_verdict = payload.get("gate_verdict")
    if isinstance(gate_verdict, dict):
        ratio = _safe_float(gate_verdict.get("train_total_ratio"))
        nds = _safe_float(gate_verdict.get("nds"))
        if ratio is not None and nds is not None:
            return f"train_total_ratio `{ratio:.4f}` and NDS `{nds:.4f}` did not clear the gate"
    return "the latest gate artifact remains blocked"


def _payload_nds(payload: dict[str, Any]) -> float | None:
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict):
        nds = _safe_float(evaluation.get("nd_score"))
        if nds is not None:
            return nds
    selected_record = payload.get("selected_record")
    if isinstance(selected_record, dict):
        evaluation = selected_record.get("evaluation")
        if isinstance(evaluation, dict):
            return _safe_float(evaluation.get("nd_score"))
    return None


def _payload_mean_ap(payload: dict[str, Any]) -> float | None:
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict):
        mean_ap = _safe_float(evaluation.get("mean_ap"))
        if mean_ap is not None:
            return mean_ap
    selected_record = payload.get("selected_record")
    if isinstance(selected_record, dict):
        evaluation = selected_record.get("evaluation")
        if isinstance(evaluation, dict):
            return _safe_float(evaluation.get("mean_ap"))
    return None


def _payload_train_total_ratio(payload: dict[str, Any]) -> float | None:
    gate_verdict = payload.get("gate_verdict")
    if isinstance(gate_verdict, dict):
        return _safe_float(gate_verdict.get("train_total_ratio"))
    return None


def _payload_car_ap_4m(payload: dict[str, Any]) -> float | None:
    gate_verdict = payload.get("gate_verdict")
    if isinstance(gate_verdict, dict):
        return _safe_float(gate_verdict.get("car_ap_4m"))
    return None


def _semantic_citation(item: dict[str, Any], repo_root: Path) -> str:
    source_path = _as_opt_str(item.get("source_path"))
    if source_path is None:
        return "uncited semantic hit"
    return _repo_link(repo_root / source_path)


def safe_sync_research_memory(
    repo_root: Path = REPO_ROOT,
    *,
    config: ResearchMemoryConfig | None = None,
) -> dict[str, Any]:
    """Best-effort memory sync that never raises into the main research loop."""

    try:
        return sync_research_memory(repo_root, config=config)
    except Exception as exc:  # pragma: no cover - defensive by design.
        return {
            "status": "error",
            "error": repr(exc),
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        }


def safe_build_research_brief(
    repo_root: Path = REPO_ROOT,
    *,
    config: ResearchMemoryConfig | None = None,
    persist_log: bool = False,
) -> dict[str, Any]:
    """Best-effort brief generation that never raises into the main research loop."""

    try:
        return build_research_brief(repo_root, config=config, persist_log=persist_log).to_dict()
    except Exception as exc:  # pragma: no cover - defensive by design.
        return {
            "status": "error",
            "error": repr(exc),
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        }
