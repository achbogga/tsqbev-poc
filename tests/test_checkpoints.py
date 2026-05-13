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


def test_load_model_checkpoint_ignores_optional_teacher_seed_mismatch(tmp_path) -> None:
    config = ModelConfig.small()
    teacher_config = config.model_copy(update={"teacher_seed_mode": "replace_lidar"})
    teacher_model = TSQBEVModel(teacher_config)
    checkpoint_path = tmp_path / "teacher_checkpoint.pt"

    payload = {
        "epoch": 1,
        "model_config": config.model_dump(),
        "model_state_dict": teacher_model.state_dict(),
        "history": [],
    }
    torch.save(payload, checkpoint_path)

    restored_model, loaded_payload = load_model_from_checkpoint(checkpoint_path)
    assert isinstance(restored_model, TSQBEVModel)
    assert loaded_payload["epoch"] == 1


def test_load_model_checkpoint_ignores_legacy_lane_attention_projection_mismatch(
    tmp_path,
) -> None:
    config = ModelConfig.small()
    model = TSQBEVModel(config)
    checkpoint_path = tmp_path / "legacy_lane_attention.pt"
    state_dict = model.state_dict()
    state_dict["core.lane_head.attn.in_proj_weight"] = torch.randn(
        config.model_dim * 3,
        config.model_dim,
    )
    state_dict["core.lane_head.attn.in_proj_bias"] = torch.randn(config.model_dim * 3)
    for key in (
        "core.lane_head.attn.q_proj.weight",
        "core.lane_head.attn.q_proj.bias",
        "core.lane_head.attn.k_proj.weight",
        "core.lane_head.attn.k_proj.bias",
        "core.lane_head.attn.v_proj.weight",
        "core.lane_head.attn.v_proj.bias",
    ):
        del state_dict[key]

    payload = {
        "epoch": 1,
        "model_config": config.model_dump(),
        "model_state_dict": state_dict,
        "history": [],
    }
    torch.save(payload, checkpoint_path)

    restored_model, loaded_payload = load_model_from_checkpoint(checkpoint_path)
    assert isinstance(restored_model, TSQBEVModel)
    assert loaded_payload["epoch"] == 1


def test_load_model_checkpoint_merges_missing_fields_from_default_config(tmp_path) -> None:
    default_config = ModelConfig.small().model_copy(
        update={
            "sam2_repo_root": "/opt/sam2",
            "sam2_model_cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2_checkpoint": "/opt/sam2/sam2.1_hiera_base_plus.pt",
            "sam2_region_prior_mode": "off",
            "sam2_region_prior_weight": 0.0,
        }
    )
    checkpoint_config = default_config.model_dump()
    del checkpoint_config["sam2_repo_root"]
    del checkpoint_config["sam2_model_cfg"]
    del checkpoint_config["sam2_checkpoint"]
    model = TSQBEVModel(default_config)
    checkpoint_path = tmp_path / "dinov3_missing_sam2.pt"

    payload = {
        "epoch": 1,
        "model_config": checkpoint_config,
        "model_state_dict": model.state_dict(),
        "history": [],
    }
    torch.save(payload, checkpoint_path)

    restored_model, _ = load_model_from_checkpoint(
        checkpoint_path,
        default_config=default_config,
    )

    assert restored_model.config.sam2_repo_root == default_config.sam2_repo_root
    assert restored_model.config.sam2_model_cfg == default_config.sam2_model_cfg
    assert restored_model.config.sam2_checkpoint == default_config.sam2_checkpoint
