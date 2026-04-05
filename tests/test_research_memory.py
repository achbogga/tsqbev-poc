from __future__ import annotations

import json
from pathlib import Path

from tsqbev.research_memory import (
    QdrantEvidenceIndex,
    ResearchMemoryConfig,
    _artifact_files,
    _collect_knowledge_facts,
    _Embedder,
    _evidence_source_files,
    _semantic_rank_key,
    build_research_brief,
    check_research_memory_health,
    query_research_memory,
    sync_research_memory,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_repo_fixture(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    _write(
        repo_root / "README.md",
        (
            "# tsqbev-poc\n\nCurrent reset target is a foundation-teacher "
            "perspective-sparse student.\n"
        ),
    )
    _write(repo_root / "program.md", "Status: enabled.\n")
    _write(repo_root / "AGENTS.md", "# Agent Rules\n\nUse memory-first research.\n")
    _write(
        repo_root / "docs" / "plan.md",
        "# Plan\n\nScale-up is blocked until quality improves.\n",
    )
    _write(
        repo_root / "docs" / "steering.md",
        "# Steering\n\nPrioritize foundation-teacher perspective-sparse reset evidence.\n",
    )
    _write(
        repo_root / "specs" / "004-research-loop-contract.md",
        "# Spec 004\n\nThe loop must remain append-only.\n",
    )

    results_record = {
        "run_id": 4,
        "recipe": "mini_propheavy_mbv3_frozen_query_boost",
        "stage": "exploit",
        "parent_recipe": "mini_propheavy_mbv3_frozen",
        "status": "completed",
        "interim_decision": "advance",
        "final_decision": "promote",
        "git_sha": "deadbee",
        "hypothesis": "query boost should improve recall",
        "mutation_reason": "bounded sparse budget increase",
        "root_cause_verdict": "optimization bottleneck",
        "config": {
            "image_backbone": "mobilenet_v3_large",
            "teacher_seed_mode": "off",
            "views": 6,
        },
        "evaluation": {"nd_score": 0.0158, "mean_ap": 0.0001},
        "val": {"total": 20.1352},
        "benchmark": {"mean_ms": 17.19},
    }
    _write(
        repo_root / "artifacts" / "research_v3" / "research_loop" / "results.jsonl",
        json.dumps(results_record) + "\n",
    )

    summary = {
        "status": "completed",
        "reference_workflow": "karpathy/autoresearch",
        "selected_recipe": "mini_propheavy_mbv3_frozen_query_boost",
        "selected_record": {
            "recipe": "mini_propheavy_mbv3_frozen_query_boost",
            "config": {
                "image_backbone": "mobilenet_v3_large",
                "teacher_seed_mode": "off",
                "views": 6,
            },
            "evaluation": {"nd_score": 0.0158, "mean_ap": 0.0001},
            "val": {"total": 20.1352},
        },
        "scale_gate_verdict": {
            "authorized": False,
            "reason": "optimization gate remains blocked by weak mini generalization",
        },
        "recommended_next_steps": [
            (
                "Continue the foundation-teacher reset and compare against reproduced "
                "BEVFusion and Sparse4D ceilings."
            ),
            "Do not scale compute until the gate is cleared.",
        ],
    }
    _write(
        repo_root / "artifacts" / "research_v3" / "research_loop" / "summary.json",
        json.dumps(summary, indent=2),
    )

    overfit_v5 = {
        "subset_size": 32,
        "evaluation": {"nd_score": 0.1479, "mean_ap": 0.1815},
        "train": {"final_val_total": 12.1011},
        "gate_verdict": {
            "passed": False,
            "train_total_ratio": 0.3447,
            "nds": 0.1479,
            "mean_ap": 0.1815,
            "car_ap_4m": 0.0,
            "nonzero_classes": 7,
        },
    }
    _write(
        repo_root
        / "artifacts"
        / "gates"
        / "recovery_v5_teacher_anchor_hypothesis"
        / "overfit_gate"
        / "summary.json",
        json.dumps(overfit_v5, indent=2),
    )

    overfit_v6 = {
        "subset_size": 32,
        "evaluation": {"nd_score": 0.1001, "mean_ap": 0.1391},
        "train": {"final_val_total": 11.9342},
        "gate_verdict": {
            "passed": False,
            "train_total_ratio": 0.4703,
            "nds": 0.1001,
            "mean_ap": 0.1391,
            "car_ap_4m": 0.5327,
            "nonzero_classes": 7,
        },
    }
    _write(
        repo_root
        / "artifacts"
        / "gates"
        / "recovery_v6_teacher_anchor_balanced"
        / "overfit_gate"
        / "summary.json",
        json.dumps(overfit_v6, indent=2),
    )

    overfit_v12 = {
        "subset_size": 32,
        "evaluation": {"nd_score": 0.0944, "mean_ap": 0.1339},
        "train": {"final_val_total": 13.2811},
        "gate_verdict": {
            "passed": False,
            "train_total_ratio": 0.3639,
            "nds": 0.0944,
            "mean_ap": 0.1339,
            "car_ap_4m": 0.3603,
            "nonzero_classes": 6,
        },
    }
    _write(
        repo_root
        / "artifacts"
        / "gates"
        / "recovery_v12_teacher_anchor_seeded_boot12_zero"
        / "overfit_gate"
        / "summary.json",
        json.dumps(overfit_v12, indent=2),
    )

    overfit_v14 = {
        "subset_size": 32,
        "evaluation": {"nd_score": 0.1553, "mean_ap": 0.1992},
        "train": {"final_val_total": 13.6891},
        "gate_verdict": {
            "passed": True,
            "train_total_ratio": 0.3624,
            "nds": 0.1553,
            "mean_ap": 0.1992,
            "car_ap_4m": 0.4958,
            "nonzero_classes": 7,
        },
    }
    _write(
        repo_root
        / "artifacts"
        / "gates"
        / "recovery_v14_teacher_anchor_quality_focal"
        / "overfit_gate"
        / "summary.json",
        json.dumps(overfit_v14, indent=2),
    )

    upstream = {
        "checkpoint_path": "pretrained/bevfusion-det.pth",
        "config_rel": (
            "configs/nuscenes/det/transfusion/secfpn/"
            "camera+lidar/swint_v0p075/convfuser.yaml"
        ),
        "headline_metrics": {
            "mAP": 0.6730,
            "NDS": 0.7072,
            "mATE": 0.2859,
            "mASE": 0.2559,
        },
        "status": "success",
        "upstream_repo_root": "/home/achbogga/projects/bevfusion",
    }
    _write(
        repo_root / "artifacts" / "bevfusion_repro" / "bevfusion_bbox_summary.json",
        json.dumps(upstream, indent=2),
    )
    _write(repo_root / "docs" / "reports" / "current.md", "# stale generated report\n")
    _write(
        repo_root / "artifacts" / "memory" / "brief.json",
        json.dumps({"stale": True}, indent=2),
    )
    _write(
        repo_root / "research" / "knowledge" / "mit_han_test.json",
        json.dumps(
            {
                "collection": "mit_han_efficient_ml",
                "entries": [
                    {
                        "id": "awq",
                        "title": "AWQ",
                        "year": 2024,
                        "area": "llm_quantization",
                        "technique_family": ["quantization", "system_co_design"],
                        "primary_bottleneck": "weight bandwidth",
                        "core_trick": (
                            "Protect activation-salient weights during low-bit weight-only "
                            "quantization instead of quantizing every channel uniformly."
                        ),
                        "apply_when": [
                            (
                                "weights dominate memory traffic and activations can stay "
                                "higher precision"
                            )
                        ],
                        "avoid_when": [
                            (
                                "activation quantization is the main bottleneck and "
                                "weight-only compression is insufficient"
                            )
                        ],
                        "tsqbev_actions": [
                            "use for offline teacher compression before cluster-scale distillation"
                        ],
                        "sources": {
                            "project": "https://hanlab.mit.edu/projects/tinyml",
                            "code": "https://github.com/mit-han-lab/llm-awq",
                        },
                    }
                ],
            },
            indent=2,
        ),
    )
    return repo_root


def test_sync_research_memory_writes_catalog_and_manifest(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    config = ResearchMemoryConfig(
        memory_root=tmp_path / ".local" / "memory",
        artifact_root=repo_root / "artifacts" / "memory",
        reports_root=repo_root / "docs" / "reports",
        report_log_root=repo_root / "docs" / "reports" / "log",
        steering_path=repo_root / "docs" / "steering.md",
        qdrant_enabled=False,
        mem0_enabled=False,
    )

    summary = sync_research_memory(repo_root, config=config)

    assert summary["event_count"] >= 3
    assert summary["evidence_count"] > 0
    assert summary["fact_count"] >= 3
    assert summary["build_id"]
    assert Path(summary["catalog_path"]).exists()
    assert "builds" in summary["catalog_path"]
    assert (config.artifact_root / "sync_manifest.json").exists()


def test_sync_research_memory_promotes_current_build_manifest(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    config = ResearchMemoryConfig(
        memory_root=tmp_path / ".local" / "memory",
        artifact_root=repo_root / "artifacts" / "memory",
        reports_root=repo_root / "docs" / "reports",
        report_log_root=repo_root / "docs" / "reports" / "log",
        steering_path=repo_root / "docs" / "steering.md",
        qdrant_enabled=False,
        mem0_enabled=False,
    )

    summary = sync_research_memory(repo_root, config=config)
    health = check_research_memory_health(repo_root, config=config)

    assert summary["current_build"]["catalog_path"].endswith("catalog.duckdb")
    assert config.current_build_manifest_path.exists()
    assert config.artifact_current_build_manifest_path.exists()
    assert config.current_build_link.is_symlink()
    assert health["current_build"] is not None
    assert health["catalog"]["exists"] is True


def test_sync_research_memory_promotes_even_when_semantic_backends_raise(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import tsqbev.research_memory as research_memory

    class _FakeEmbedder:
        reranker_enabled = False
        reranker_provider = "none"
        reranker_reason = None
        reranker_fallback_reason = None

    class _BoomQdrant:
        def __init__(self, config: ResearchMemoryConfig) -> None:
            self.enabled = True
            self.mode = "server"
            self.reason = None
            self.embedder_provider = "fastembed"
            self._embedder = _FakeEmbedder()

        def upsert_chunks(self, chunks: list[object]) -> int:
            raise RuntimeError("qdrant boom")

    class _BoomMem0:
        def __init__(self, config: ResearchMemoryConfig) -> None:
            raise RuntimeError("mem0 boom")

    monkeypatch.setattr(research_memory, "QdrantEvidenceIndex", _BoomQdrant)
    monkeypatch.setattr(research_memory, "Mem0MemoryBackend", _BoomMem0)

    repo_root = _make_repo_fixture(tmp_path)
    config = ResearchMemoryConfig(
        memory_root=tmp_path / ".local" / "memory",
        artifact_root=repo_root / "artifacts" / "memory",
        reports_root=repo_root / "docs" / "reports",
        report_log_root=repo_root / "docs" / "reports" / "log",
        steering_path=repo_root / "docs" / "steering.md",
        qdrant_enabled=True,
        mem0_enabled=True,
    )

    summary = sync_research_memory(repo_root, config=config)

    assert Path(summary["catalog_path"]).exists()
    assert summary["current_build"]["catalog_path"].endswith("catalog.duckdb")
    assert summary["qdrant"]["indexed_chunks"] == 0
    assert "qdrant boom" in str(summary["qdrant"]["reason"])
    assert "mem0 boom" in str(summary["mem0"]["reason"])
    assert summary["mem0"]["spool"]["remaining"] == summary["fact_count"]
    assert "upsert_events" in summary["phase_timings_s"]
    assert "qdrant_sync" in summary["phase_timings_s"]
    assert (config.artifact_root / "sync_manifest.json").exists()


def test_evidence_source_files_exclude_results_jsonl(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)

    event_files = {path.relative_to(repo_root).as_posix() for path in _artifact_files(repo_root)}
    evidence_files = {
        path.relative_to(repo_root).as_posix() for path in _evidence_source_files(repo_root)
    }

    assert "artifacts/research_v3/research_loop/results.jsonl" in event_files
    assert "artifacts/research_v3/research_loop/results.jsonl" not in evidence_files
    assert "artifacts/research_v3/research_loop/summary.json" not in evidence_files
    assert "docs/plan.md" in evidence_files


def test_build_research_brief_uses_indexed_state(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    config = ResearchMemoryConfig(
        memory_root=tmp_path / ".local" / "memory",
        artifact_root=repo_root / "artifacts" / "memory",
        reports_root=repo_root / "docs" / "reports",
        report_log_root=repo_root / "docs" / "reports" / "log",
        steering_path=repo_root / "docs" / "steering.md",
        qdrant_enabled=False,
        mem0_enabled=False,
    )
    sync_research_memory(repo_root, config=config)

    brief = build_research_brief(repo_root, config=config, persist_log=True)

    assert any("mini_propheavy_mbv3_frozen_query_boost" in line for line in brief.current_state)
    assert any(
        "recovery_v14_teacher_anchor_quality_focal" in line for line in brief.current_state
    )
    assert all(
        "recovery_v12_teacher_anchor_seeded_boot12_zero" not in line
        for line in brief.current_state
    )
    assert all("recovery_v5_teacher_anchor_hypothesis" not in line for line in brief.current_state)
    assert any("bevfusion:convfuser" in line for line in brief.current_state)
    assert any(
        "mini_train -> mini_val" in line and "recovery_v14_teacher_anchor_quality_focal" in line
        for line in brief.open_blockers
    )


def test_semantic_rank_key_prefers_rerank_score() -> None:
    low_semantic_high_rerank = {
        "source_path": "docs/plan.md",
        "semantic_score": 0.10,
        "rerank_score": 0.95,
    }
    high_semantic_no_rerank = {
        "source_path": "docs/plan.md",
        "semantic_score": 0.80,
    }

    assert _semantic_rank_key(low_semantic_high_rerank) > _semantic_rank_key(
        high_semantic_no_rerank
    )


def test_embedder_can_use_optional_cohere_reranker(monkeypatch) -> None:
    import tsqbev.research_memory as research_memory

    class _FakeResult:
        def __init__(self, index: int, relevance_score: float) -> None:
            self.index = index
            self.relevance_score = relevance_score

    class _FakeResponse:
        def __init__(self) -> None:
            self.results = [_FakeResult(1, 0.9), _FakeResult(0, 0.2)]

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def rerank(
            self,
            *,
            model: str,
            query: str,
            documents: list[str],
            top_n: int,
        ) -> _FakeResponse:
            assert model == "rerank-v4.0-pro"
            assert query == "why is scale-up blocked?"
            assert top_n == 2
            assert documents == ["doc a", "doc b"]
            return _FakeResponse()

    monkeypatch.setattr(research_memory, "CohereClientV2", _FakeClient)

    embedder = _Embedder(
        ResearchMemoryConfig(
            reranker_enabled=True,
            reranker_provider="cohere",
            cohere_api_key="test-key",
            qdrant_enabled=False,
            mem0_enabled=False,
        )
    )

    assert embedder.reranker_enabled is True
    assert embedder.reranker_provider == "cohere"
    reranked = embedder.rerank(
        "why is scale-up blocked?",
        [{"text": "doc a"}, {"text": "doc b"}],
    )
    assert reranked[0]["text"] == "doc b"
    assert reranked[0]["rerank_score"] == 0.9


def test_embedder_falls_back_to_fastembed_when_cohere_runtime_fails(monkeypatch) -> None:
    import tsqbev.research_memory as research_memory

    class _FailingClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def rerank(
            self,
            *,
            model: str,
            query: str,
            documents: list[str],
            top_n: int,
        ) -> object:
            raise RuntimeError("rate limited")

    class _FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
            assert query == "why is scale-up blocked?"
            assert texts == ["doc a", "doc b"]
            return [(1, 0.8), (0, 0.1)]

    monkeypatch.setattr(research_memory, "CohereClientV2", _FailingClient)
    monkeypatch.setattr(research_memory, "TextCrossEncoder", _FakeCrossEncoder)

    embedder = _Embedder(
        ResearchMemoryConfig(
            reranker_enabled=True,
            reranker_provider="cohere",
            cohere_api_key="test-key",
            qdrant_enabled=False,
            mem0_enabled=False,
        )
    )

    reranked = embedder.rerank(
        "why is scale-up blocked?",
        [{"text": "doc a"}, {"text": "doc b"}],
    )

    assert embedder.reranker_provider == "fastembed"
    assert "fell back to fastembed" in str(embedder.reranker_reason)
    assert reranked[0]["text"] == "doc b"


def test_qdrant_index_degrades_cleanly_on_upsert_failure(monkeypatch) -> None:
    import tsqbev.research_memory as research_memory

    class _FakeCollections:
        collections: list[object] = []

    class _FakeClient:
        def get_collections(self) -> _FakeCollections:
            return _FakeCollections()

        def create_collection(self, **kwargs: object) -> None:
            return None

        def upsert(self, **kwargs: object) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(research_memory, "_http_ok", lambda _url: True)
    monkeypatch.setattr(research_memory, "QdrantClient", lambda **kwargs: _FakeClient())

    index = QdrantEvidenceIndex(
        ResearchMemoryConfig(qdrant_enabled=True, qdrant_mode="server", mem0_enabled=False)
    )
    assert index.enabled is True

    from tsqbev.research_memory import EvidenceChunk

    count = index.upsert_chunks(
        [
            EvidenceChunk(
                chunk_id="chunk-1",
                source_path="docs/plan.md",
                kind="doc",
                title="Plan",
                text="scale blocker evidence",
                citation="docs/plan.md",
                payload={},
            )
        ]
    )
    assert count == 0
    assert index.enabled is False
    assert index.reason is not None


def test_qdrant_index_upserts_in_batches(monkeypatch) -> None:
    import tsqbev.research_memory as research_memory

    class _FakeCollections:
        collections: list[object] = []

    class _FakeClient:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def get_collections(self) -> _FakeCollections:
            return _FakeCollections()

        def create_collection(self, **kwargs: object) -> None:
            return None

        def upsert(self, **kwargs: object) -> None:
            points = kwargs["points"]
            self.calls.append(len(points))

    fake_client = _FakeClient()

    class _FakeEmbedder:
        def __init__(self, config: ResearchMemoryConfig) -> None:
            self.provider = "fastembed"
            self.dimension = 3
            self.reranker_enabled = False
            self.reranker_provider = "none"
            self.reranker_reason = None
            self.reranker_fallback_reason = None

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(research_memory, "_http_ok", lambda _url: True)
    monkeypatch.setattr(research_memory, "QdrantClient", lambda **kwargs: fake_client)
    monkeypatch.setattr(research_memory, "_Embedder", _FakeEmbedder)

    index = QdrantEvidenceIndex(
        ResearchMemoryConfig(
            qdrant_enabled=True,
            qdrant_mode="server",
            qdrant_upsert_batch_size=2,
            mem0_enabled=False,
        )
    )

    from tsqbev.research_memory import EvidenceChunk

    count = index.upsert_chunks(
        [
            EvidenceChunk(
                chunk_id=f"chunk-{idx}",
                source_path="docs/plan.md",
                kind="doc",
                title=f"Plan {idx}",
                text=f"evidence {idx}",
                citation="docs/plan.md",
                payload={},
            )
            for idx in range(5)
        ]
    )

    assert count == 5
    assert fake_client.calls == [2, 2, 1]


def test_query_research_memory_returns_exact_facts(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    config = ResearchMemoryConfig(
        memory_root=tmp_path / ".local" / "memory",
        artifact_root=repo_root / "artifacts" / "memory",
        reports_root=repo_root / "docs" / "reports",
        report_log_root=repo_root / "docs" / "reports" / "log",
        steering_path=repo_root / "docs" / "steering.md",
        qdrant_enabled=False,
        mem0_enabled=False,
    )
    sync_research_memory(repo_root, config=config)

    result = query_research_memory("scale-up blocked", repo_root=repo_root, config=config, limit=4)

    assert result["exact_facts"]
    assert any("Scale-up is still blocked" in item["claim"] for item in result["exact_facts"])


def test_build_research_brief_prefers_scale_blocker_for_current_incumbent(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    _write(
        repo_root / "artifacts" / "research_v16" / "research_loop" / "results.jsonl",
        json.dumps(
            {
                "run_id": 2,
                "recipe": "carryover_recovery_v14_teacher_anchor_quality_focal_query_boost",
                "stage": "exploit",
                "status": "completed",
                "final_decision": "promote",
                "git_sha": "feedbee",
                "config": {
                    "image_backbone": "mobilenet_v3_large",
                    "teacher_seed_mode": "replace_lidar",
                },
                "evaluation": {
                    "nd_score": 0.1491,
                    "mean_ap": 0.1848,
                    "label_aps": {"car": {"4.0": 0.6079}},
                    "tp_errors": {"trans_err": 0.6705},
                },
                "val": {"total": 9.9361},
                "benchmark": {"mean_ms": 18.39},
            }
        )
        + "\n",
    )
    _write(
        repo_root / "artifacts" / "research_v16" / "research_loop" / "summary.json",
        json.dumps(
            {
                "status": "completed",
                "reference_workflow": "karpathy/autoresearch",
                "selected_recipe": (
                    "carryover_recovery_v14_teacher_anchor_quality_focal_query_boost"
                ),
                "selected_record": {
                    "recipe": "carryover_recovery_v14_teacher_anchor_quality_focal_query_boost",
                    "config": {
                        "image_backbone": "mobilenet_v3_large",
                        "teacher_seed_mode": "replace_lidar",
                    },
                    "evaluation": {
                        "nd_score": 0.1491,
                        "mean_ap": 0.1848,
                        "label_aps": {"car": {"4.0": 0.6079}},
                        "tp_errors": {"trans_err": 0.6705},
                    },
                    "val": {"total": 9.9361},
                    "benchmark": {"mean_ms": 18.39},
                },
                "scale_gate_verdict": {
                    "authorized": False,
                    "reason": "at least one scale gate remains unmet; do not spend 10x compute yet",
                    "gates": {
                        "geometry_sanity": {"passed": False},
                        "source_mix_stability": {"passed": False},
                    },
                },
                "recommended_next_steps": [
                    "run a paired teacher-on versus teacher-off mini invocation",
                    "reduce exported boxes per sample",
                ],
            },
            indent=2,
        ),
    )

    config = ResearchMemoryConfig(
        memory_root=tmp_path / ".local" / "memory",
        artifact_root=repo_root / "artifacts" / "memory",
        reports_root=repo_root / "docs" / "reports",
        report_log_root=repo_root / "docs" / "reports" / "log",
        steering_path=repo_root / "docs" / "steering.md",
        qdrant_enabled=False,
        mem0_enabled=False,
    )
    sync_research_memory(repo_root, config=config)

    brief = build_research_brief(repo_root, config=config, persist_log=False)

    assert any(
        "carryover_recovery_v14_teacher_anchor_quality_focal_query_boost" in line
        for line in brief.current_state
    )
    assert any(
        "artifacts/research_v16/research_loop/summary.json" in line
        for line in brief.open_blockers
    )
    assert all(
        "recovery_v12_teacher_anchor_seeded_boot12_zero" not in line
        for line in brief.open_blockers
    )


def test_artifact_files_exclude_generated_memory_outputs(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)

    files = {path.relative_to(repo_root).as_posix() for path in _artifact_files(repo_root)}

    assert "docs/reports/current.md" not in files
    assert "artifacts/memory/brief.json" not in files
    assert "artifacts/bevfusion_repro/bevfusion_bbox_summary.json" in files
    assert "research/knowledge/mit_han_test.json" in files


def test_collect_knowledge_facts_ingests_structured_literature_notes(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)

    facts = _collect_knowledge_facts(repo_root)

    assert any(fact.kind == "literature_note" for fact in facts)
    assert any("AWQ" in fact.claim for fact in facts)


def test_query_research_memory_returns_knowledge_facts(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    config = ResearchMemoryConfig(
        memory_root=tmp_path / ".local" / "memory",
        artifact_root=repo_root / "artifacts" / "memory",
        reports_root=repo_root / "docs" / "reports",
        report_log_root=repo_root / "docs" / "reports" / "log",
        steering_path=repo_root / "docs" / "steering.md",
        qdrant_enabled=False,
        mem0_enabled=False,
    )
    sync_research_memory(repo_root, config=config)

    result = query_research_memory(
        "AWQ activation-salient weights", repo_root=repo_root, config=config
    )

    assert any("AWQ" in item["claim"] for item in result["exact_facts"])
