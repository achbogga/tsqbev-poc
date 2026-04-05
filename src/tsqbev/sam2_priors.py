"""Local SAM 2.1 prior provider integration.

References:
- SAM 2.1 official repo:
  https://github.com/facebookresearch/sam2
- SAM 2.1 checkpoint download docs:
  https://github.com/facebookresearch/sam2#download-checkpoints
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tsqbev.config import ModelConfig
from tsqbev.contracts import CameraProposals, MapPriorBatch

Tensor = torch.Tensor


def _ensure_repo_on_path(repo_root: Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _load_sam2_predictor(
    repo_root: Path,
    model_cfg: str,
    checkpoint: Path,
    *,
    device: str,
) -> Any:
    if not repo_root.exists():
        raise FileNotFoundError(f"SAM2 repo root `{repo_root}` does not exist")
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint `{checkpoint}` does not exist")
    _ensure_repo_on_path(repo_root)
    build_sam = importlib.import_module("sam2.build_sam")
    predictor_mod = importlib.import_module("sam2.sam2_image_predictor")
    sam_model = build_sam.build_sam2(
        config_file=model_cfg,
        ckpt_path=str(checkpoint),
        device=device,
        mode="eval",
        apply_postprocessing=True,
    )
    return predictor_mod.SAM2ImagePredictor(sam_model)


def _image_to_uint8(image: Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("SAM2 images must be a single RGB tensor in CHW format")
    rgb = image.detach().float().cpu()
    if float(rgb.max().item()) <= 1.5 and float(rgb.min().item()) >= -0.5:
        rgb = rgb.clamp(0.0, 1.0) * 255.0
    else:
        rgb = rgb.clamp(0.0, 255.0)
    return rgb.permute(1, 2, 0).round().to(torch.uint8).numpy()


def _expand_prior_features(raw: Tensor, dim: int) -> Tensor:
    pieces = [
        raw,
        raw * raw,
        torch.sin(raw * torch.pi),
        torch.cos(raw * torch.pi),
    ]
    expanded = torch.cat(pieces, dim=-1)
    if expanded.shape[-1] < dim:
        repeat = (dim + expanded.shape[-1] - 1) // expanded.shape[-1]
        expanded = expanded.repeat_interleave(repeat, dim=-1)
    return expanded[..., :dim]


def _box_feature_vector(
    box: Tensor,
    score: Tensor,
    iou: Tensor,
    mask_prob: Tensor,
    proposal_rank: int,
    proposal_count: int,
    image_height: int,
    image_width: int,
) -> tuple[Tensor, Tensor]:
    score = score.reshape(())
    iou = iou.reshape(())
    x0, y0, x1, y1 = box.unbind(-1)
    width = (x1 - x0).clamp_min(1.0)
    height = (y1 - y0).clamp_min(1.0)
    cx = x0 + 0.5 * width
    cy = y0 + 0.5 * height
    mask_mass = mask_prob.sum().clamp_min(1.0)
    ys = torch.linspace(0.0, 1.0, mask_prob.shape[-2], device=mask_prob.device)
    xs = torch.linspace(0.0, 1.0, mask_prob.shape[-1], device=mask_prob.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    centroid_x = ((mask_prob * grid_x).sum() / mask_mass).clamp(0.0, 1.0)
    centroid_y = ((mask_prob * grid_y).sum() / mask_mass).clamp(0.0, 1.0)
    area_frac = ((width / image_width) * (height / image_height)).clamp_min(0.0)
    aspect_ratio = (width / height).clamp(0.0, 8.0)
    rank_norm = torch.tensor(
        float(proposal_rank) / max(1.0, float(proposal_count - 1)),
        dtype=box.dtype,
        device=box.device,
    )
    raw = torch.stack(
        (
            cx / float(image_width),
            cy / float(image_height),
            width / float(image_width),
            height / float(image_height),
            area_frac,
            score.clamp(0.0, 1.0),
            iou.clamp(0.0, 1.0),
            mask_prob.mean().clamp(0.0, 1.0),
            mask_prob.std(unbiased=False).clamp(0.0, 1.0),
            centroid_x,
            centroid_y,
            aspect_ratio / 8.0,
            rank_norm,
        )
    )
    coords_xy = torch.stack((cx / float(image_width), cy / float(image_height)))
    return raw, coords_xy


@dataclass(slots=True)
class SAM2PriorProvider:
    """Instantiate SAM 2.1 locally and turn proposal boxes into map priors."""

    repo_root: Path
    model_cfg: str
    checkpoint: Path
    region_prior_weight: float
    map_input_dim: int
    device: str = "cpu"
    _predictor: Any | None = None

    def validate(self) -> dict[str, str]:
        self._ensure_predictor()
        return {
            "repo_root": str(self.repo_root),
            "model_cfg": self.model_cfg,
            "checkpoint": str(self.checkpoint),
            "device": self.device,
            "status": "ready",
        }

    def _ensure_predictor(self) -> Any:
        if self._predictor is None:
            self._predictor = _load_sam2_predictor(
                self.repo_root,
                self.model_cfg,
                self.checkpoint,
                device=self.device,
            )
        return self._predictor

    def build_map_priors(
        self,
        images: Tensor,
        proposals: CameraProposals,
    ) -> MapPriorBatch:
        if images.ndim != 5:
            raise ValueError("SAM2 prior provider expects images with rank 5")
        batch, views = images.shape[:2]
        proposals.validate(batch, views)
        predictor = self._ensure_predictor()
        flat_images = [_image_to_uint8(images[b, v]) for b in range(batch) for v in range(views)]
        flat_boxes = [
            proposals.boxes_xyxy[b, v].detach().float().cpu().numpy()
            for b in range(batch)
            for v in range(views)
        ]
        predictor.set_image_batch(flat_images)
        masks_batch, iou_batch, _ = predictor.predict_batch(
            box_batch=flat_boxes,
            multimask_output=False,
            return_logits=True,
            normalize_coords=True,
        )

        token_count = proposals.boxes_xyxy.shape[2]
        total_tokens = views * token_count
        tokens = torch.zeros(
            batch,
            total_tokens,
            self.map_input_dim,
            dtype=images.dtype,
            device=images.device,
        )
        coords_xy = torch.zeros(
            batch,
            total_tokens,
            2,
            dtype=images.dtype,
            device=images.device,
        )
        valid_mask = torch.zeros(
            batch,
            total_tokens,
            dtype=torch.bool,
            device=images.device,
        )

        for batch_index in range(batch):
            view_token_offset = 0
            for view_index in range(views):
                flat_index = batch_index * views + view_index
                view_boxes = proposals.boxes_xyxy[batch_index, view_index]
                view_scores = proposals.scores[batch_index, view_index]
                view_masks = torch.as_tensor(
                    np.asarray(masks_batch[flat_index]),
                    dtype=images.dtype,
                    device=images.device,
                )
                view_ious = torch.as_tensor(
                    np.asarray(iou_batch[flat_index]),
                    dtype=images.dtype,
                    device=images.device,
                )
                if view_masks.ndim == 2:
                    view_masks = view_masks.unsqueeze(0)
                if view_ious.ndim == 0:
                    view_ious = view_ious.unsqueeze(0)
                for proposal_index in range(token_count):
                    mask_logits = view_masks[min(proposal_index, view_masks.shape[0] - 1)]
                    mask_prob = torch.sigmoid(mask_logits)
                    raw, coords = _box_feature_vector(
                        view_boxes[proposal_index],
                        view_scores[proposal_index],
                        view_ious[min(proposal_index, view_ious.shape[0] - 1)],
                        mask_prob,
                        proposal_index,
                        token_count,
                        image_height=int(images.shape[-2]),
                        image_width=int(images.shape[-1]),
                    )
                    token = _expand_prior_features(raw.to(images.device), self.map_input_dim)
                    token = token * float(self.region_prior_weight)
                    slot = view_token_offset + proposal_index
                    tokens[batch_index, slot] = token
                    coords_xy[batch_index, slot] = coords.to(images.device)
                    valid_mask[batch_index, slot] = True
                view_token_offset += token_count

        return MapPriorBatch(tokens=tokens, coords_xy=coords_xy, valid_mask=valid_mask)


def build_sam2_prior_provider(
    config: ModelConfig,
    *,
    device: str | None = None,
) -> SAM2PriorProvider | None:
    if config.sam2_region_prior_mode == "off":
        return None
    if config.sam2_region_prior_mode != "proposal_boxes":
        raise ValueError(f"unsupported SAM2 region prior mode: {config.sam2_region_prior_mode}")
    if config.sam2_repo_root is None:
        return None
    if config.sam2_model_cfg is None:
        return None
    if config.sam2_checkpoint is None:
        return None
    provider = SAM2PriorProvider(
        repo_root=Path(config.sam2_repo_root),
        model_cfg=config.sam2_model_cfg,
        checkpoint=Path(config.sam2_checkpoint),
        region_prior_weight=config.sam2_region_prior_weight,
        map_input_dim=config.map_input_dim,
        device=device or "cpu",
    )
    provider.validate()
    return provider


def validate_local_sam2_assets(config: ModelConfig, *, device: str | None = None) -> dict[str, str]:
    if config.sam2_region_prior_mode == "off":
        raise ValueError("SAM2 region priors are disabled in the current config")
    if config.sam2_repo_root is None:
        raise ValueError("sam2_region_prior_mode requires `sam2_repo_root`")
    if config.sam2_model_cfg is None:
        raise ValueError("sam2_region_prior_mode requires `sam2_model_cfg`")
    if config.sam2_checkpoint is None:
        raise ValueError("sam2_region_prior_mode requires `sam2_checkpoint`")
    provider = build_sam2_prior_provider(config, device=device)
    if provider is None:
        raise ValueError("SAM2 prior provider failed to initialize")
    return provider.validate()
