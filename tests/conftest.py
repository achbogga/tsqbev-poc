from __future__ import annotations

import pytest

from tsqbev.config import ModelConfig
from tsqbev.synthetic import make_synthetic_batch


@pytest.fixture()
def small_config() -> ModelConfig:
    return ModelConfig.small()


@pytest.fixture()
def synthetic_batch(small_config: ModelConfig):
    return make_synthetic_batch(small_config, batch_size=2)
