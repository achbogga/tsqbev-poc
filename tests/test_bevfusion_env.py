from __future__ import annotations

from pathlib import Path

from tsqbev.bevfusion_env import (
    bevfusion_official_commands,
    check_bevfusion_environment,
    render_bevfusion_runbook_markdown,
)


def _touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_bevfusion_env_reports_missing_info_pickles_as_note(tmp_path: Path) -> None:
    repo_root = tmp_path / "bevfusion"
    dataset_root = tmp_path / "nuscenes"

    _touch(repo_root, "docker/Dockerfile")
    _touch(repo_root, "tools/download_pretrained.sh")
    _touch(repo_root, "tools/create_data.py")
    _touch(
        repo_root,
        "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
    )

    (dataset_root / "samples").mkdir(parents=True)
    (dataset_root / "sweeps").mkdir(parents=True)
    (dataset_root / "maps").mkdir(parents=True)
    (dataset_root / "maps" / "expansion").mkdir(parents=True)
    (dataset_root / "maps" / "basemap").mkdir(parents=True)
    (dataset_root / "maps" / "prediction").mkdir(parents=True)
    (dataset_root / "v1.0-trainval").mkdir(parents=True)
    (dataset_root / "v1.0-mini").mkdir(parents=True)

    status = check_bevfusion_environment(repo_root=repo_root, dataset_root=dataset_root)

    assert status.samples_present is True
    assert status.train_info_present is False
    assert any("ann files" in note for note in status.notes)
    assert "official BEVFusion Dockerfile missing" not in status.blockers


def test_bevfusion_env_notes_missing_map_expansion(tmp_path: Path) -> None:
    repo_root = tmp_path / "bevfusion"
    dataset_root = tmp_path / "nuscenes"

    _touch(repo_root, "docker/Dockerfile")
    _touch(repo_root, "tools/download_pretrained.sh")
    _touch(repo_root, "tools/create_data.py")
    _touch(
        repo_root,
        "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
    )
    (dataset_root / "samples").mkdir(parents=True)
    (dataset_root / "sweeps").mkdir(parents=True)
    (dataset_root / "maps").mkdir(parents=True)
    (dataset_root / "v1.0-trainval").mkdir(parents=True)

    status = check_bevfusion_environment(repo_root=repo_root, dataset_root=dataset_root)

    assert "nuScenes map expansion missing under maps/expansion" in status.blockers
    assert "nuScenes map expansion missing under maps/basemap" in status.blockers
    assert "nuScenes map expansion missing under maps/prediction" in status.blockers
    assert any("shared nuScenes pipeline loads NuScenesMap" in note for note in status.notes)


def test_bevfusion_env_flags_missing_repo_and_dataset(tmp_path: Path) -> None:
    status = check_bevfusion_environment(
        repo_root=tmp_path / "missing_bevfusion",
        dataset_root=tmp_path / "missing_nuscenes",
    )

    assert status.recommended_strategy in {"official-docker", "blocked"}
    assert "BEVFusion repo root missing" in status.blockers
    assert "nuScenes dataset root missing" in status.blockers


def test_bevfusion_official_commands_embed_repo_and_dataset_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "bevfusion"
    dataset_root = tmp_path / "nuscenes"
    commands = bevfusion_official_commands(repo_root=repo_root, dataset_root=dataset_root)

    assert "bootstrap_bevfusion_official.sh" in commands["build_image"]
    assert "run_bevfusion_nuscenes_prep.sh" in commands["prepare_nuscenes"]
    assert str(repo_root.resolve()) in commands["prepare_nuscenes"]
    assert str(dataset_root.resolve()) in commands["evaluate_detection"]
    assert "run_bevfusion_nuscenes_eval.sh" in commands["evaluate_segmentation"]


def test_bevfusion_runbook_mentions_helper_and_exact_ann_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "bevfusion"
    dataset_root = tmp_path / "nuscenes"

    report = render_bevfusion_runbook_markdown(repo_root=repo_root, dataset_root=dataset_root)

    assert "run_bevfusion_nuscenes_prep.sh" in report
    assert "bootstrap_bevfusion_official.sh" in report
    assert "run_bevfusion_nuscenes_eval.sh" in report
