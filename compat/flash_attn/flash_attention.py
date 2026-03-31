"""Compatibility shim for archived BEVFusion import paths.

Primary sources:
- BEVFusion radar encoder import path:
  https://github.com/mit-han-lab/bevfusion/blob/main/mmdet3d/models/backbones/radar_encoder.py
- BEVFusion camera+lidar detection config, which does not use radar:
  https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/default.yaml
"""

from __future__ import annotations


class FlashMHA:
    """Guardrail shim.

    The published camera+lidar BEVFusion path imports `radar_encoder` eagerly through
    `mmdet3d.models.backbones.__init__`, but the selected config has `use_radar: false`.
    This shim allows non-radar configs to import cleanly while still failing loudly if a
    radar path tries to instantiate FlashMHA.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "flash_attn compatibility shim was imported. The selected BEVFusion camera+lidar "
            "baseline should not instantiate RadarEncoder or FlashMHA."
        )
