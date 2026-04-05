from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tsqbev.config import ModelConfig
from tsqbev.contracts import CameraProposals
from tsqbev.sam2_priors import (
    SAM2PriorProvider,
    build_sam2_prior_provider,
    validate_local_sam2_assets,
)


class _FakeSam2Predictor:
    def __init__(self) -> None:
        self.images: list[np.ndarray] | None = None

    def set_image_batch(self, image_list) -> None:  # type: ignore[no-untyped-def]
        self.images = list(image_list)

    def predict_batch(
        self,
        *,
        box_batch,
        multimask_output,
        return_logits,
        normalize_coords,
    ):
        del multimask_output, return_logits, normalize_coords
        masks = []
        ious = []
        low_res = []
        for boxes in box_batch:
            count = int(np.asarray(boxes).shape[0])
            masks.append(np.ones((count, 1, 8, 8), dtype=np.float32))
            ious.append(np.full((count, 1), 0.75, dtype=np.float32))
            low_res.append(np.ones((count, 1, 8, 8), dtype=np.float32))
        return masks, ious, low_res


def _sam2_config(tmp_path: Path) -> ModelConfig:
    repo_root = tmp_path / "sam2"
    repo_root.mkdir()
    model_cfg = repo_root / "configs" / "sam2.1"
    model_cfg.mkdir(parents=True)
    checkpoint = tmp_path / "sam2.1_hiera_base_plus.pt"
    checkpoint.touch()
    return ModelConfig.small().model_copy(
        update={
            "sam2_repo_root": str(repo_root),
            "sam2_model_cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2_checkpoint": str(checkpoint),
            "sam2_region_prior_mode": "proposal_boxes",
            "sam2_region_prior_weight": 0.25,
            "map_input_dim": 128,
        }
    )


def test_build_sam2_prior_provider_requires_proposal_mode(tmp_path: Path) -> None:
    config = ModelConfig.small()
    assert build_sam2_prior_provider(config) is None


def test_validate_local_sam2_assets_reports_ready(monkeypatch, tmp_path: Path) -> None:
    config = _sam2_config(tmp_path)

    def fake_loader(repo_root, model_cfg, checkpoint, *, device):  # type: ignore[no-untyped-def]
        assert Path(repo_root).exists()
        assert Path(checkpoint).exists()
        assert model_cfg.endswith("sam2.1_hiera_b+.yaml")
        assert device == "cpu"
        return _FakeSam2Predictor()

    monkeypatch.setattr("tsqbev.sam2_priors._load_sam2_predictor", fake_loader)

    report = validate_local_sam2_assets(config, device="cpu")

    assert report["status"] == "ready"
    assert report["checkpoint"].endswith("sam2.1_hiera_base_plus.pt")


def test_sam2_prior_provider_builds_map_priors_from_proposals(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _sam2_config(tmp_path)

    def fake_loader(repo_root, model_cfg, checkpoint, *, device):  # type: ignore[no-untyped-def]
        del repo_root, model_cfg, checkpoint, device
        return _FakeSam2Predictor()

    monkeypatch.setattr("tsqbev.sam2_priors._load_sam2_predictor", fake_loader)

    provider = SAM2PriorProvider(
        repo_root=Path(config.sam2_repo_root or ""),
        model_cfg=str(config.sam2_model_cfg),
        checkpoint=Path(config.sam2_checkpoint or ""),
        region_prior_weight=config.sam2_region_prior_weight,
        map_input_dim=config.map_input_dim,
        device="cpu",
    )

    images = torch.rand(1, 2, 3, 16, 16)
    proposals = CameraProposals(
        boxes_xyxy=torch.tensor(
            [[[[1.0, 1.0, 8.0, 8.0], [2.0, 2.0, 10.0, 10.0], [3.0, 3.0, 12.0, 12.0]],
              [[1.0, 1.0, 8.0, 8.0], [2.0, 2.0, 10.0, 10.0], [3.0, 3.0, 12.0, 12.0]]]]
        ),
        scores=torch.ones(1, 2, 3),
    )

    map_priors = provider.build_map_priors(images, proposals)

    assert map_priors.tokens.shape == (1, 6, 128)
    assert map_priors.coords_xy.shape == (1, 6, 2)
    assert map_priors.valid_mask.shape == (1, 6)
    assert bool(map_priors.valid_mask.all())
    assert torch.count_nonzero(map_priors.tokens) > 0
