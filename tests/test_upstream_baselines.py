from __future__ import annotations

from pathlib import Path

from tsqbev.upstream_baselines import local_upstream_baselines, upstream_baselines


def _touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_upstream_baseline_manifest_contains_core_reset_baselines() -> None:
    manifest = {baseline.key: baseline for baseline in upstream_baselines()}

    assert "openpcdet_centerpoint_pointpillar_nuscenes" in manifest
    assert "bevfusion_detection_nuscenes" in manifest
    assert "bevdet_r50_4d_depth_cbgs_nuscenes" in manifest
    assert "maptrv2_nusc_r50_24ep" in manifest


def test_local_upstream_baselines_resolve_config_paths(tmp_path: Path) -> None:
    maptr_root = tmp_path / "MapTR"
    _touch(maptr_root, "projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py")
    _touch(maptr_root, "mmdetection3d/.keep")
    _touch(maptr_root, "tools/.keep")

    bevdet_root = tmp_path / "BEVDet"
    _touch(bevdet_root, "configs/bevdet/bevdet-r50-4d-depth-cbgs.py")
    _touch(bevdet_root, "mmdet3d/.keep")
    _touch(bevdet_root, "tools/.keep")

    statuses = {status.key: status for status in local_upstream_baselines(tmp_path)}

    assert statuses["bevdet_r50_4d_depth_cbgs_nuscenes"].config_present is True
    assert statuses["maptrv2_nusc_r50_24ep"].config_path.endswith("maptrv2_nusc_r50_24ep.py")
