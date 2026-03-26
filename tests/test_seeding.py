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
