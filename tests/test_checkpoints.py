from __future__ import annotations

import torch

from tsqbev.checkpoints import load_model_from_checkpoint, save_model_checkpoint
from tsqbev.config import ModelConfig
from tsqbev.model import TSQBEVModel


def test_save_and_load_model_checkpoint_round_trip(tmp_path) -> None:
    config = ModelConfig.small()
    model = TSQBEVModel(config)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_model_checkpoint(
        model,
        config,
        checkpoint_path,
        epoch=3,
        history=[{"epoch": 3, "train": {"total": 1.0}, "val": {"total": 0.5}}],
    )

    restored_model, payload = load_model_from_checkpoint(checkpoint_path)
    assert isinstance(restored_model, TSQBEVModel)
    assert payload["epoch"] == 3
    assert payload["model_config"]["model_dim"] == config.model_dim

    for expected, actual in zip(
        model.state_dict().values(),
        restored_model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(expected, actual)
