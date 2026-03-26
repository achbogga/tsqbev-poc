"""Latency predictor and tier helpers.

References:
- HotBEV hardware-aware latency predictor:
  https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf
"""

from __future__ import annotations

from dataclasses import dataclass

from tsqbev.config import LatencyPredictorConfig, ModelConfig


@dataclass(slots=True)
class LatencyFeatures:
    """Features used by the simple latency gate."""

    params_m: float
    sample_ops_m: float
    lidar_pillars_k: float
    temporal_frames: float
    activations_mb: float


class LatencyPredictor:
    """A simple predictor used for triage, not as ground truth."""

    def __init__(self, config: LatencyPredictorConfig | None = None) -> None:
        self.config = config or LatencyPredictorConfig()

    def predict_ms(self, features: LatencyFeatures) -> float:
        cfg = self.config
        return (
            cfg.b0
            + cfg.b1 * features.params_m
            + cfg.b2 * features.sample_ops_m
            + cfg.b3 * features.lidar_pillars_k
            + cfg.b4 * features.temporal_frames
            + cfg.b5 * features.activations_mb
        )

    def production_pass(self, p95_ms: float) -> bool:
        return p95_ms <= self.config.production_p95_ms

    def stretch_pass(self, p95_ms: float) -> bool:
        return p95_ms <= self.config.stretch_p95_ms


def features_from_config(
    config: ModelConfig, params_m: float, active_pillars: float, activations_mb: float
) -> LatencyFeatures:
    """Build latency features from a config and measured/precomputed stats."""

    sample_ops_m = (
        config.views
        * config.feature_levels
        * (config.max_object_queries + config.lane_queries)
        * config.sample_keypoints
    ) / 1_000_000.0
    return LatencyFeatures(
        params_m=params_m,
        sample_ops_m=sample_ops_m,
        lidar_pillars_k=active_pillars,
        temporal_frames=float(config.temporal_frames),
        activations_mb=activations_mb,
    )
