"""Checkpoint helpers for reproducible training and evaluation.

References:
- Karpathy autoresearch workflow staging:
  https://github.com/karpathy/autoresearch
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tsqbev.config import ModelConfig
from tsqbev.model import TSQBEVModel


def _allowed_checkpoint_key_mismatch(key: str) -> bool:
    """Allow additive optional teacher bootstrap modules across checkpoint versions."""

    if key.startswith("teacher_seed_encoder."):
        return True
    lane_attention_prefixes = (
        "core.lane_head.attn.in_proj_weight",
        "core.lane_head.attn.in_proj_bias",
        "core.lane_head.attn.q_proj.",
        "core.lane_head.attn.k_proj.",
        "core.lane_head.attn.v_proj.",
    )
    return key.startswith(lane_attention_prefixes)


def save_model_checkpoint(
    model: TSQBEVModel,
    config: ModelConfig,
    checkpoint_path: str | Path,
    *,
    epoch: int,
    history: list[dict[str, object]],
) -> Path:
    """Write a checkpoint with both weights and the resolved model config."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_config": config.model_dump(),
        "model_state_dict": model.state_dict(),
        "history": history,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint_payload(
    checkpoint_path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Read a training checkpoint payload from disk."""

    payload = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint at {checkpoint_path} must contain a dict payload")
    return payload


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    default_config: ModelConfig | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[TSQBEVModel, dict[str, Any]]:
    """Construct a model from a checkpoint payload and load its weights."""

    payload = load_checkpoint_payload(checkpoint_path, map_location=map_location)
    config_data = payload.get("model_config")
    if config_data is None:
        if default_config is None:
            raise KeyError("checkpoint is missing model_config and no default_config was provided")
        config = default_config
    else:
        if default_config is None:
            config = ModelConfig.model_validate(config_data)
        else:
            merged_config = default_config.model_dump()
            if not isinstance(config_data, dict):
                raise TypeError("checkpoint model_config must be a dict")
            merged_config.update(config_data)
            config = ModelConfig.model_validate(merged_config)
    model = TSQBEVModel(config)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint model_state_dict must be a dict")
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = [
        key for key in incompatible.unexpected_keys if not _allowed_checkpoint_key_mismatch(key)
    ]
    missing = [
        key for key in incompatible.missing_keys if not _allowed_checkpoint_key_mismatch(key)
    ]
    if unexpected or missing:
        raise RuntimeError(
            "checkpoint state_dict mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model, payload


def load_weights_into_model_from_checkpoint(
    model: TSQBEVModel,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> list[str]:
    """Load checkpoint weights into an existing model."""

    payload = load_checkpoint_payload(checkpoint_path, map_location=map_location)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint model_state_dict must be a dict")
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = [
        key for key in incompatible.unexpected_keys if not _allowed_checkpoint_key_mismatch(key)
    ]
    missing = [
        key for key in incompatible.missing_keys if not _allowed_checkpoint_key_mismatch(key)
    ]
    if unexpected or missing:
        raise RuntimeError(
            "checkpoint state_dict mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return list(incompatible.missing_keys)
