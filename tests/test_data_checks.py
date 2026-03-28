from __future__ import annotations

from tsqbev.data_checks import check_nuscenes_root, check_openlane_root


def test_check_nuscenes_root_reports_ready(tmp_path) -> None:
    (tmp_path / "v1.0-mini").mkdir()
    for filename in (
        "scene.json",
        "sample_data.json",
        "calibrated_sensor.json",
        "ego_pose.json",
    ):
        (tmp_path / "v1.0-mini" / filename).write_text("{}")
    (tmp_path / "samples" / "CAM_FRONT").mkdir(parents=True)
    (tmp_path / "samples" / "LIDAR_TOP").mkdir(parents=True)
    (tmp_path / "maps").mkdir()

    report = check_nuscenes_root(tmp_path)
    assert report["ready"] is True
    assert report["metadata_version"] == "v1.0-mini"


def test_check_openlane_root_accepts_lane3d_300_or_1000(tmp_path) -> None:
    (tmp_path / "images" / "training").mkdir(parents=True)
    (tmp_path / "images" / "validation").mkdir(parents=True)
    (tmp_path / "lane3d_300" / "training").mkdir(parents=True)
    (tmp_path / "lane3d_300" / "validation").mkdir(parents=True)

    report = check_openlane_root(tmp_path)
    assert report["ready"] is True
