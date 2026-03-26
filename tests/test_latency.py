from __future__ import annotations

from tsqbev.config import LatencyPredictorConfig, ModelConfig
from tsqbev.latency import LatencyPredictor, features_from_config


def test_latency_predictor_uses_all_feature_terms() -> None:
    predictor = LatencyPredictor(
        LatencyPredictorConfig(b0=1.0, b1=2.0, b2=3.0, b3=4.0, b4=5.0, b5=6.0)
    )
    config = ModelConfig.small()
    features = features_from_config(config, params_m=1.0, active_pillars=2.0, activations_mb=3.0)
    expected = (
        1.0
        + 2.0 * features.params_m
        + 3.0 * features.sample_ops_m
        + 4.0 * features.lidar_pillars_k
        + 5.0 * features.temporal_frames
        + 6.0 * features.activations_mb
    )
    assert predictor.predict_ms(features) == expected
