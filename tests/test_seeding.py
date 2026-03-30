from __future__ import annotations

import torch

from tsqbev.config import ModelConfig
from tsqbev.seeding import TriSourceQueryRouter


def test_tri_source_router_keeps_requested_query_budget(small_config: ModelConfig) -> None:
    router = TriSourceQueryRouter(small_config)
    batch = 2
    lidar_queries = torch.randn(batch, small_config.q_lidar, small_config.model_dim)
    lidar_refs = torch.randn(batch, small_config.q_lidar, 3)
    lidar_scores = torch.rand(batch, small_config.q_lidar)
    proposal_queries = torch.randn(batch, small_config.q_2d, small_config.model_dim)
    proposal_refs = torch.randn(batch, small_config.q_2d, 3)
    proposal_scores = torch.rand(batch, small_config.q_2d)
    global_queries = torch.randn(batch, small_config.q_global, small_config.model_dim)
    global_refs = torch.randn(batch, small_config.q_global, 3)
    global_scores = torch.rand(batch, small_config.q_global)

    seed_bank = router(
        lidar_queries,
        lidar_refs,
        lidar_scores,
        proposal_queries,
        proposal_refs,
        proposal_scores,
        global_queries,
        global_refs,
        global_scores,
    )
    seed_bank.validate(batch)
    assert seed_bank.embeddings.shape[1] == small_config.max_object_queries


def test_tri_source_router_preserves_source_diversity_when_lidar_scores_dominate(
    small_config: ModelConfig,
) -> None:
    router = TriSourceQueryRouter(small_config)
    batch = 1
    lidar_queries = torch.randn(batch, small_config.q_lidar, small_config.model_dim)
    lidar_refs = torch.randn(batch, small_config.q_lidar, 3)
    lidar_scores = torch.full((batch, small_config.q_lidar), 1000.0)
    proposal_queries = torch.randn(batch, small_config.q_2d, small_config.model_dim)
    proposal_refs = torch.randn(batch, small_config.q_2d, 3)
    proposal_scores = torch.linspace(0.1, 0.9, small_config.q_2d).unsqueeze(0)
    global_queries = torch.randn(batch, small_config.q_global, small_config.model_dim)
    global_refs = torch.randn(batch, small_config.q_global, 3)
    global_scores = torch.full((batch, small_config.q_global), 0.5)

    seed_bank = router(
        lidar_queries,
        lidar_refs,
        lidar_scores,
        proposal_queries,
        proposal_refs,
        proposal_scores,
        global_queries,
        global_refs,
        global_scores,
    )

    source_ids = seed_bank.source_ids[0]
    unique_sources = set(source_ids.tolist())
    assert unique_sources == {
        TriSourceQueryRouter.SOURCE_LIDAR,
        TriSourceQueryRouter.SOURCE_PROPOSAL,
        TriSourceQueryRouter.SOURCE_GLOBAL,
    }
    assert float(seed_bank.scores.min()) >= 0.0
    assert float(seed_bank.scores.max()) <= 1.0


def test_anchor_first_router_prefers_lidar_anchors_when_available(
    small_config: ModelConfig,
) -> None:
    config = small_config.model_copy(update={"router_mode": "anchor_first"})
    router = TriSourceQueryRouter(config)
    batch = 1
    lidar_queries = torch.randn(batch, config.q_lidar, config.model_dim)
    lidar_refs = torch.randn(batch, config.q_lidar, 3)
    lidar_scores = torch.linspace(0.2, 1.0, config.q_lidar).unsqueeze(0)
    proposal_queries = torch.randn(batch, config.q_2d, config.model_dim)
    proposal_refs = torch.randn(batch, config.q_2d, 3)
    proposal_scores = torch.ones(batch, config.q_2d)
    global_queries = torch.randn(batch, config.q_global, config.model_dim)
    global_refs = torch.randn(batch, config.q_global, 3)
    global_scores = torch.ones(batch, config.q_global)

    seed_bank = router(
        lidar_queries,
        lidar_refs,
        lidar_scores,
        proposal_queries,
        proposal_refs,
        proposal_scores,
        global_queries,
        global_refs,
        global_scores,
    )

    assert torch.all(seed_bank.source_ids == TriSourceQueryRouter.SOURCE_LIDAR)
