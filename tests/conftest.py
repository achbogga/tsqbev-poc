from __future__ import annotations

import pytest

from tsqbev.config import ModelConfig
from tsqbev.synthetic import make_synthetic_batch


@pytest.fixture(autouse=True)
def disable_wandb_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSQBEV_WANDB", "0")
    monkeypatch.setenv("TSQBEV_MEMORY_ENABLED", "0")


@pytest.fixture()
def small_config() -> ModelConfig:
    return ModelConfig.small()


@pytest.fixture()
def synthetic_batch(small_config: ModelConfig):
    return make_synthetic_batch(small_config, batch_size=2)
