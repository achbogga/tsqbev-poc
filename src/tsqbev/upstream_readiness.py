"""Readiness checks for the public upstream repos in the dense-BEV reset plan.

References:
- OpenPCDet:
  https://github.com/open-mmlab/OpenPCDet
- BEVFusion:
  https://github.com/mit-han-lab/bevfusion
- BEVDet:
  https://github.com/HuangJunJie2017/BEVDet
- MapTR:
  https://github.com/hustvl/MapTR
- PersFormer:
  https://github.com/OpenDriveLab/PersFormer_3DLane
- DINOv2:
  https://github.com/facebookresearch/dinov2
- DINOv3:
  https://github.com/facebookresearch/dinov3
- EfficientViT:
  https://github.com/mit-han-lab/efficientvit
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UpstreamRepoStatus:
    """Local readiness status for one external repo."""

    key: str
    expected_root: str
    present: bool
    missing_files: tuple[str, ...]
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_REPO_EXPECTATIONS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "openpcdet",
        "OpenPCDet_official",
        ("pcdet/__init__.py", "tools/cfgs", "requirements.txt"),
        "Primary LiDAR runtime and teacher substrate.",
    ),
    (
        "bevfusion",
        "bevfusion",
        ("configs", "mmdet3d", "tools"),
        "Primary shared-BEV fusion trunk candidate.",
    ),
    (
        "bevdet",
        "BEVDet",
        ("configs", "mmdet3d", "tools"),
        "Primary temporal camera BEV encoder candidate.",
    ),
    (
        "maptr",
        "MapTR",
        ("projects", "mmdetection3d", "tools"),
        "Primary vector map / lane head candidate.",
    ),
    (
        "persformer",
        "PersFormer_3DLane",
        ("models", "experiments", "main_train_GenLaneNet_ext.py"),
        "Auxiliary OpenLane lane teacher/eval repo.",
    ),
    (
        "dinov2",
        "dinov2",
        ("hubconf.py", "README.md", "dinov2"),
        "Dense visual feature teacher.",
    ),
    (
        "dinov3",
        "dinov3",
        ("hubconf.py", "README.md", "dinov3"),
        "Next-step dense visual feature teacher.",
    ),
    (
        "efficientvit",
        "efficientvit",
        ("README.md", "efficientvit", "applications"),
        "Phase-2 camera efficiency specialization path.",
    ),
)


def check_upstream_stack(projects_root: Path) -> tuple[UpstreamRepoStatus, ...]:
    """Check whether expected upstream repos are present under the projects root."""

    statuses: list[UpstreamRepoStatus] = []
    for key, folder, required_files, note in _REPO_EXPECTATIONS:
        root = projects_root / folder
        missing = tuple(path for path in required_files if not (root / path).exists())
        statuses.append(
            UpstreamRepoStatus(
                key=key,
                expected_root=str(root),
                present=root.exists() and not missing,
                missing_files=missing,
                note=note,
            )
        )
    return tuple(statuses)
