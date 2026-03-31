"""Pinned upstream baseline manifest for the dense-BEV reset stack.

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
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from tsqbev.upstream_readiness import check_upstream_stack


@dataclass(frozen=True, slots=True)
class UpstreamBaseline:
    """One pinned public baseline recipe for reproduction."""

    key: str
    repo_key: str
    task: str
    expected_branch: str | None
    config_relpath: str
    weights_url: str | None
    reported_metrics: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocalBaselineStatus:
    """Local status for one pinned upstream baseline."""

    key: str
    repo_key: str
    config_path: str
    config_present: bool
    repo_present: bool
    expected_branch: str | None
    actual_branch: str | None
    head_sha: str | None
    blockers: tuple[str, ...]
    reported_metrics: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_BASELINES: tuple[UpstreamBaseline, ...] = (
    UpstreamBaseline(
        key="openpcdet_centerpoint_pointpillar_nuscenes",
        repo_key="openpcdet",
        task="lidar_detection",
        expected_branch=None,
        config_relpath="tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml",
        weights_url="https://github.com/open-mmlab/OpenPCDet",
        reported_metrics=("mAP 50.03", "NDS 60.70"),
        rationale="Default LiDAR runtime and teacher baseline.",
    ),
    UpstreamBaseline(
        key="bevfusion_detection_nuscenes",
        repo_key="bevfusion",
        task="multimodal_detection",
        expected_branch=None,
        config_relpath="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
        weights_url="https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1",
        reported_metrics=("mAP 68.52", "NDS 71.38"),
        rationale="Primary public multimodal detection checkpoint.",
    ),
    UpstreamBaseline(
        key="bevfusion_segmentation_nuscenes",
        repo_key="bevfusion",
        task="bev_map_segmentation",
        expected_branch=None,
        config_relpath="configs/nuscenes/seg/fusion-bev256d2-lss.yaml",
        weights_url="https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9&dl=1",
        reported_metrics=("mIoU 62.95",),
        rationale="Primary shared-BEV map segmentation checkpoint.",
    ),
    UpstreamBaseline(
        key="bevdet_r50_4d_depth_cbgs_nuscenes",
        repo_key="bevdet",
        task="camera_detection",
        expected_branch=None,
        config_relpath="configs/bevdet/bevdet-r50-4d-depth-cbgs.py",
        weights_url="https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1",
        reported_metrics=("mAP 36.1", "NDS 48.3", "FPS 25.2"),
        rationale="Primary temporal camera BEV detection baseline.",
    ),
    UpstreamBaseline(
        key="maptrv2_nusc_r50_24ep",
        repo_key="maptr",
        task="vector_map",
        expected_branch="maptrv2",
        config_relpath="projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py",
        weights_url="https://drive.google.com/file/d/1AmQ3fT-J-MM4B8kh_9Gm2G5guM92Agww/view?usp=sharing",
        reported_metrics=("mAP 61.4", "FPS 14.1"),
        rationale="Primary vector lane/map baseline for the dense-BEV reset.",
    ),
    UpstreamBaseline(
        key="persformer_openlane",
        repo_key="persformer",
        task="lane_detection",
        expected_branch=None,
        config_relpath="config/persformer_openlane.py",
        weights_url=None,
        reported_metrics=("OpenLane F-score baseline documented in README",),
        rationale="Auxiliary OpenLane lane transfer and evaluation control arm.",
    ),
)


_LOCAL_REPO_DIRS: dict[str, str] = {
    "openpcdet": "OpenPCDet_official",
    "bevfusion": "bevfusion",
    "bevdet": "BEVDet",
    "maptr": "MapTR",
    "persformer": "PersFormer_3DLane",
}


def upstream_baselines() -> tuple[UpstreamBaseline, ...]:
    """Return the pinned public baseline manifest."""

    return _BASELINES


def local_upstream_baselines(projects_root: Path) -> tuple[LocalBaselineStatus, ...]:
    """Resolve baseline configs against the local clone layout and branch state."""

    readiness = {status.key: status for status in check_upstream_stack(projects_root)}
    resolved: list[LocalBaselineStatus] = []
    for baseline in upstream_baselines():
        repo_dir = _LOCAL_REPO_DIRS[baseline.repo_key]
        config_path = projects_root / repo_dir / baseline.config_relpath
        repo_status = readiness[baseline.repo_key]
        resolved.append(
            LocalBaselineStatus(
                key=baseline.key,
                repo_key=baseline.repo_key,
                config_path=str(config_path),
                config_present=config_path.exists(),
                repo_present=repo_status.present,
                expected_branch=baseline.expected_branch,
                actual_branch=repo_status.actual_branch,
                head_sha=repo_status.head_sha,
                blockers=repo_status.blockers,
                reported_metrics=baseline.reported_metrics,
            )
        )
    return tuple(resolved)
