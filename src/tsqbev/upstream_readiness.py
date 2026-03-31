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

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UpstreamRepoStatus:
    """Local readiness status for one external repo."""

    key: str
    expected_root: str
    present: bool
    missing_files: tuple[str, ...]
    expected_branch: str | None
    actual_branch: str | None
    head_sha: str | None
    blockers: tuple[str, ...]
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_REPO_EXPECTATIONS: tuple[tuple[str, str, tuple[str, ...], str | None, str], ...] = (
    (
        "openpcdet",
        "OpenPCDet_official",
        ("pcdet/__init__.py", "tools/cfgs", "requirements.txt"),
        None,
        "Primary LiDAR runtime and teacher substrate.",
    ),
    (
        "bevfusion",
        "bevfusion",
        ("configs", "mmdet3d", "tools"),
        None,
        "Primary shared-BEV fusion trunk candidate.",
    ),
    (
        "bevdet",
        "BEVDet",
        ("configs", "mmdet3d", "tools"),
        None,
        "Primary temporal camera BEV encoder candidate.",
    ),
    (
        "maptr",
        "MapTR",
        ("projects", "mmdetection3d", "tools"),
        "maptrv2",
        "Primary vector map / lane head candidate.",
    ),
    (
        "persformer",
        "PersFormer_3DLane",
        ("models", "experiments", "main_persformer.py"),
        None,
        "Auxiliary OpenLane lane teacher/eval repo.",
    ),
    (
        "dinov2",
        "dinov2",
        ("hubconf.py", "README.md", "dinov2"),
        None,
        "Dense visual feature teacher.",
    ),
    (
        "dinov3",
        "dinov3",
        ("hubconf.py", "README.md", "dinov3"),
        None,
        "Next-step dense visual feature teacher.",
    ),
    (
        "efficientvit",
        "efficientvit",
        ("README.md", "efficientvit", "applications"),
        None,
        "Phase-2 camera efficiency specialization path.",
    ),
)


def _git_output(repo_root: Path, *args: str) -> str | None:
    if not (repo_root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def check_upstream_stack(projects_root: Path) -> tuple[UpstreamRepoStatus, ...]:
    """Check whether expected upstream repos are present under the projects root."""

    statuses: list[UpstreamRepoStatus] = []
    for key, folder, required_files, expected_branch, note in _REPO_EXPECTATIONS:
        root = projects_root / folder
        missing = tuple(path for path in required_files if not (root / path).exists())
        actual_branch = _git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
        head_sha = _git_output(root, "rev-parse", "HEAD")
        blockers: list[str] = []
        if not root.exists():
            blockers.append("repo root missing")
        if missing:
            blockers.append("required files missing")
        if expected_branch is not None and actual_branch != expected_branch:
            blockers.append(
                f"expected branch {expected_branch}, found {actual_branch or 'unknown'}"
            )
        statuses.append(
            UpstreamRepoStatus(
                key=key,
                expected_root=str(root),
                present=root.exists() and not missing and not blockers,
                missing_files=missing,
                expected_branch=expected_branch,
                actual_branch=actual_branch,
                head_sha=head_sha,
                blockers=tuple(blockers),
                note=note,
            )
        )
    return tuple(statuses)
