"""ONNX export helpers for the TSQBEV core.

References:
- TensorRT export and deployment guidance in HotBEV:
  https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from tsqbev.config import ModelConfig
from tsqbev.model import TSQBEVCore

Tensor = torch.Tensor

EXPORT_INPUT_NAMES = (
    "images",
    "intrinsics",
    "extrinsics",
    "lidar_queries",
    "lidar_refs",
    "lidar_scores",
    "proposal_queries",
    "proposal_refs",
    "proposal_scores",
)


class ExportableCore(nn.Module):
    """Tensor-only wrapper around the core network."""

    def __init__(self, core: TSQBEVCore) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        images: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        lidar_queries: Tensor,
        lidar_refs: Tensor,
        lidar_scores: Tensor,
        proposal_queries: Tensor,
        proposal_refs: Tensor,
        proposal_scores: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        outputs = self.core(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            lidar_queries=lidar_queries,
            lidar_refs=lidar_refs,
            lidar_scores=lidar_scores,
            proposal_queries=proposal_queries,
            proposal_refs=proposal_refs,
            proposal_scores=proposal_scores,
            map_priors=None,
            state=None,
        )
        return (
            outputs["object_logits"],
            outputs["object_boxes"],
            outputs["lane_logits"],
            outputs["lane_polylines"],
        )


def build_export_inputs(
    config: ModelConfig,
    image_height: int = 96,
    image_width: int = 160,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, ...]:
    """Build fixed-size synthetic inputs for ONNX export."""

    batch = 1
    resolved_device = torch.device(device) if device is not None else None
    images = torch.randn(
        batch,
        config.views,
        3,
        image_height,
        image_width,
        device=resolved_device,
        dtype=dtype,
    )
    intrinsics = torch.eye(3, device=resolved_device, dtype=dtype).view(1, 1, 3, 3)
    intrinsics = intrinsics.repeat(batch, config.views, 1, 1)
    extrinsics = torch.eye(4, device=resolved_device, dtype=dtype).view(1, 1, 4, 4)
    extrinsics = extrinsics.repeat(batch, config.views, 1, 1)
    lidar_queries = torch.randn(
        batch,
        config.q_lidar,
        config.model_dim,
        device=resolved_device,
        dtype=dtype,
    )
    lidar_refs = torch.randn(batch, config.q_lidar, 3, device=resolved_device, dtype=dtype)
    lidar_scores = torch.rand(batch, config.q_lidar, device=resolved_device, dtype=dtype)
    proposal_queries = torch.randn(
        batch,
        config.q_2d,
        config.model_dim,
        device=resolved_device,
        dtype=dtype,
    )
    proposal_refs = torch.randn(batch, config.q_2d, 3, device=resolved_device, dtype=dtype)
    proposal_scores = torch.rand(batch, config.q_2d, device=resolved_device, dtype=dtype)
    return (
        images,
        intrinsics,
        extrinsics,
        lidar_queries,
        lidar_refs,
        lidar_scores,
        proposal_queries,
        proposal_refs,
        proposal_scores,
    )


def build_export_input_dict(
    config: ModelConfig,
    image_height: int = 96,
    image_width: int = 160,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Build a named dictionary of export tensors."""

    return dict(
        zip(
            EXPORT_INPUT_NAMES,
            build_export_inputs(
                config,
                image_height=image_height,
                image_width=image_width,
                device=device,
                dtype=dtype,
            ),
            strict=True,
        )
    )


def export_core_to_onnx(
    core: TSQBEVCore,
    config: ModelConfig,
    path: str | Path,
    image_height: int = 96,
    image_width: int = 160,
) -> Path:
    """Export the core model to ONNX using synthetic prepared inputs."""

    exportable = ExportableCore(core)
    inputs = build_export_inputs(config, image_height=image_height, image_width=image_width)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        exportable,
        inputs,
        output_path,
        input_names=list(EXPORT_INPUT_NAMES),
        output_names=["object_logits", "object_boxes", "lane_logits", "lane_polylines"],
        opset_version=17,
    )
    return output_path
