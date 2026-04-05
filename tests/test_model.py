from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from tsqbev.config import ModelConfig
from tsqbev.model import TSQBEVModel


def test_model_forward_shapes(small_config, synthetic_batch) -> None:
    model = TSQBEVModel(small_config)
    outputs = model(synthetic_batch)
    assert outputs["object_logits"].shape == (
        synthetic_batch.batch_size,
        small_config.max_object_queries,
        small_config.num_object_classes,
    )
    assert outputs["objectness_logits"].shape == (
        synthetic_batch.batch_size,
        small_config.max_object_queries,
    )
    assert outputs["object_boxes"].shape == (
        synthetic_batch.batch_size,
        small_config.max_object_queries,
        9,
    )
    assert outputs["lane_logits"].shape == (
        synthetic_batch.batch_size,
        small_config.lane_queries,
    )
    assert outputs["lane_polylines"].shape == (
        synthetic_batch.batch_size,
        small_config.lane_queries,
        small_config.lane_points,
        3,
    )


def test_model_temporal_state_round_trip(small_config, synthetic_batch) -> None:
    model = TSQBEVModel(small_config)
    first = model(synthetic_batch)
    second = model(synthetic_batch, state=first["temporal_state"])
    assert (
        second["temporal_state"].object_queries.shape
        == first["temporal_state"].object_queries.shape
    )


def test_model_bounded_centers_stay_near_seed_refs(small_config, synthetic_batch) -> None:
    model = TSQBEVModel(small_config)
    outputs = model(synthetic_batch)
    seed_refs = outputs["seed_bank"].refs_xyz
    centers = outputs["object_boxes"][..., :3]
    max_offset = (centers - seed_refs).abs().max()
    assert torch.isfinite(max_offset)
    assert float(max_offset) <= 8.0 + 1e-5
    assert torch.all(outputs["object_boxes"][..., 3:6] > 0.0)


def test_model_supports_torchvision_backbone(synthetic_batch) -> None:
    config = ModelConfig.small().model_copy(
        update={
            "image_backbone": "mobilenet_v3_large",
            "pretrained_image_backbone": False,
            "freeze_image_backbone": True,
        }
    )
    model = TSQBEVModel(config)
    batch = replace(
        synthetic_batch,
        images=synthetic_batch.images[:1],
        lidar_points=synthetic_batch.lidar_points[:1],
        lidar_mask=synthetic_batch.lidar_mask[:1],
        intrinsics=synthetic_batch.intrinsics[:1],
        extrinsics=synthetic_batch.extrinsics[:1],
        ego_pose=synthetic_batch.ego_pose[:1],
        time_delta_s=synthetic_batch.time_delta_s[:1],
        od_targets=None,
        lane_targets=None,
        teacher_targets=replace(
            synthetic_batch.teacher_targets,
            object_features=synthetic_batch.teacher_targets.object_features[:1]
            if synthetic_batch.teacher_targets is not None
            and synthetic_batch.teacher_targets.object_features is not None
            else None,
            object_boxes=synthetic_batch.teacher_targets.object_boxes[:1]
            if synthetic_batch.teacher_targets is not None
            and synthetic_batch.teacher_targets.object_boxes is not None
            else None,
            object_labels=synthetic_batch.teacher_targets.object_labels[:1]
            if synthetic_batch.teacher_targets is not None
            and synthetic_batch.teacher_targets.object_labels is not None
            else None,
            object_scores=synthetic_batch.teacher_targets.object_scores[:1]
            if synthetic_batch.teacher_targets is not None
            and synthetic_batch.teacher_targets.object_scores is not None
            else None,
            lane_features=synthetic_batch.teacher_targets.lane_features[:1]
            if synthetic_batch.teacher_targets is not None
            and synthetic_batch.teacher_targets.lane_features is not None
            else None,
            router_logits=synthetic_batch.teacher_targets.router_logits[:1]
            if synthetic_batch.teacher_targets is not None
            and synthetic_batch.teacher_targets.router_logits is not None
            else None,
            valid_mask=synthetic_batch.teacher_targets.valid_mask[:1]
            if synthetic_batch.teacher_targets is not None
            and synthetic_batch.teacher_targets.valid_mask is not None
            else None,
        )
        if synthetic_batch.teacher_targets is not None
        else None,
        map_priors=replace(
            synthetic_batch.map_priors,
            tokens=synthetic_batch.map_priors.tokens[:1],
            coords_xy=synthetic_batch.map_priors.coords_xy[:1],
            valid_mask=synthetic_batch.map_priors.valid_mask[:1],
        )
        if synthetic_batch.map_priors is not None
        else None,
    )
    outputs = model(batch)
    assert outputs["object_logits"].shape[0] == 1


def test_model_can_replace_lidar_seeds_with_teacher_targets(synthetic_batch) -> None:
    config = ModelConfig.small().model_copy(update={"teacher_seed_mode": "replace_lidar"})
    model = TSQBEVModel(config)
    outputs = model(synthetic_batch)
    seed_bank = outputs["seed_bank"]
    assert (seed_bank.source_ids == 0).any()
    assert seed_bank.prior_valid_mask is not None
    assert bool(seed_bank.prior_valid_mask.any())
    assert seed_bank.prior_labels is not None
    assert seed_bank.prior_scores is not None


def test_model_can_replace_lidar_refs_with_teacher_targets(synthetic_batch) -> None:
    assert synthetic_batch.teacher_targets is not None
    config = ModelConfig.small().model_copy(update={"teacher_seed_mode": "replace_lidar_refs"})
    model = TSQBEVModel(config)
    outputs = model(synthetic_batch)
    seed_bank = outputs["seed_bank"]
    assert seed_bank.refs_xyz.shape[1] == config.max_object_queries
    assert (seed_bank.source_ids == 0).any()


def test_model_supports_dinov2_projected_backbone(monkeypatch, synthetic_batch, tmp_path) -> None:
    class FakeDino(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = 384
            self.last_input_shape: tuple[int, ...] | None = None

        def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: list[int],
            reshape: bool = True,
            norm: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.last_input_shape = tuple(x.shape)
            batch = x.shape[0]
            height = x.shape[-2] // 14
            width = x.shape[-1] // 14
            low = torch.randn(batch, self.embed_dim, height, width, device=x.device, dtype=x.dtype)
            high = torch.randn(
                batch,
                self.embed_dim,
                height,
                width,
                device=x.device,
                dtype=x.dtype,
            )
            return low, high

    fake_dino = FakeDino()
    monkeypatch.setattr("torch.hub.load", lambda *args, **kwargs: fake_dino)
    foundation_root = tmp_path / "dinov2"
    foundation_root.mkdir()
    config = ModelConfig.small().model_copy(
        update={
            "image_backbone": "dinov2_vits14_reg",
            "pretrained_image_backbone": True,
            "freeze_image_backbone": True,
            "foundation_repo_root": str(foundation_root),
            "attention_backend": "math",
        }
    )
    model = TSQBEVModel(config)
    outputs = model(synthetic_batch)
    assert outputs["object_logits"].shape[0] == synthetic_batch.batch_size
    assert fake_dino.last_input_shape is not None
    assert fake_dino.last_input_shape[-2] % 14 == 0
    assert fake_dino.last_input_shape[-1] % 14 == 0


def test_model_supports_dinov3_projected_backbone(monkeypatch, synthetic_batch, tmp_path) -> None:
    class FakeDinoV3(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = 384
            self.last_input_shape: tuple[int, ...] | None = None

        def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: list[int],
            reshape: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.last_input_shape = tuple(x.shape)
            batch = x.shape[0]
            height = x.shape[-2] // 16
            width = x.shape[-1] // 16
            low = torch.randn(batch, self.embed_dim, height, width, device=x.device, dtype=x.dtype)
            high = torch.randn(batch, self.embed_dim, height, width, device=x.device, dtype=x.dtype)
            return low, high

    fake_dino = FakeDinoV3()
    monkeypatch.setattr("torch.hub.load", lambda *args, **kwargs: fake_dino)
    foundation_root = tmp_path / "dinov3"
    foundation_root.mkdir()
    config = ModelConfig.small().model_copy(
        update={
            "image_backbone": "dinov3_vits16",
            "pretrained_image_backbone": True,
            "freeze_image_backbone": True,
            "foundation_repo_root": str(foundation_root),
            "foundation_patch_multiple": 16,
            "attention_backend": "math",
        }
    )
    model = TSQBEVModel(config)
    outputs = model(synthetic_batch)
    assert outputs["object_logits"].shape[0] == synthetic_batch.batch_size
    assert fake_dino.last_input_shape is not None
    assert fake_dino.last_input_shape[-2] % 16 == 0
    assert fake_dino.last_input_shape[-1] % 16 == 0
