from __future__ import annotations

from pathlib import Path

from tsqbev.bevdet_env import (
    PRIMARY_BEVDET_CONFIG,
    _bevdet_probe_is_catastrophic,
    _probe_epoch_from_checkpoint,
    bevdet_official_commands,
    check_bevdet_environment,
    render_bevdet_runbook_markdown,
)


def _touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_bevdet_env_reports_missing_info_pickles_as_note(tmp_path: Path) -> None:
    repo_root = tmp_path / "BEVDet"
    dataset_root = tmp_path / "nuscenes"

    _touch(repo_root, "docker/Dockerfile")
    _touch(repo_root, "tools/create_data_bevdet.py")
    _touch(repo_root, PRIMARY_BEVDET_CONFIG)
    _touch(repo_root, "configs/bevdet/bevdet-r50-4d-depth-cbgs.py")
    (dataset_root / "samples").mkdir(parents=True)
    (dataset_root / "sweeps").mkdir(parents=True)
    (dataset_root / "maps").mkdir(parents=True)
    (dataset_root / "v1.0-trainval").mkdir(parents=True)
    (dataset_root / "v1.0-mini").mkdir(parents=True)

    status = check_bevdet_environment(repo_root=repo_root, dataset_root=dataset_root)

    assert status.samples_present is True
    assert status.mini_train_info_present is False
    assert any("mini BEVDet info PKLs are missing" in note for note in status.notes)


def test_bevdet_env_flags_missing_repo_and_dataset(tmp_path: Path) -> None:
    status = check_bevdet_environment(
        repo_root=tmp_path / "missing_bevdet",
        dataset_root=tmp_path / "missing_nuscenes",
    )

    assert status.recommended_strategy in {"official-docker", "blocked"}
    assert "BEVDet repo root missing" in status.blockers
    assert "nuScenes dataset root missing" in status.blockers


def test_bevdet_official_commands_embed_repo_and_dataset_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "BEVDet"
    dataset_root = tmp_path / "nuscenes"
    commands = bevdet_official_commands(repo_root=repo_root, dataset_root=dataset_root)

    assert "bootstrap_bevdet_official.sh" in commands["build_image"]
    assert "run_bevdet_nuscenes_prep.sh" in commands["prepare_mini_infos"]
    assert "run_bevdet_nuscenes_train.sh" in commands["run_primary_public_student"]
    assert str(repo_root.resolve()) in commands["build_image"]
    assert str(dataset_root.resolve()) in commands["prepare_trainval_infos"]


def test_bevdet_runbook_mentions_public_student_helpers(tmp_path: Path) -> None:
    repo_root = tmp_path / "BEVDet"
    dataset_root = tmp_path / "nuscenes"

    report = render_bevdet_runbook_markdown(repo_root=repo_root, dataset_root=dataset_root)

    assert "bootstrap_bevdet_official.sh" in report
    assert "run_bevdet_nuscenes_prep.sh" in report
    assert "run_bevdet_nuscenes_train.sh" in report
    assert "BEVDet Public Student Runbook" in report


def test_probe_epoch_from_checkpoint_extracts_epoch() -> None:
    assert _probe_epoch_from_checkpoint(Path("/tmp/epoch_3.pth")) == 3
    assert _probe_epoch_from_checkpoint(Path("/tmp/latest.pth")) is None


def test_bevdet_probe_catastrophic_gate_requires_zero_metrics_and_bad_geometry() -> None:
    assert _bevdet_probe_is_catastrophic(
        {"nd_score": 0.0, "mean_ap": 0.0},
        {
            "max_box_size_m": 40.0,
            "ego_translation_norm_p99": 150.0,
            "boxes_per_sample_mean": 120.0,
        },
        {"sanity_ok": 0.0, "score_mean": 0.999},
    )
    assert not _bevdet_probe_is_catastrophic(
        {"nd_score": 0.03, "mean_ap": 0.001},
        {
            "max_box_size_m": 10.0,
            "ego_translation_norm_p99": 15.0,
            "boxes_per_sample_mean": 12.0,
        },
        {"sanity_ok": 1.0, "score_mean": 0.2},
    )
