from __future__ import annotations

from pathlib import Path

from tsqbev.openpcdet_env import check_openpcdet_environment


def test_check_openpcdet_environment_reports_blockers(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "OpenPCDet"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "tools" / "cfgs" / "nuscenes_models").mkdir(parents=True)
    (repo_root / "docs" / "INSTALL.md").write_text("install")
    (repo_root / "tools" / "cfgs" / "nuscenes_models" / "cbgs_dyn_pp_centerpoint.yaml").write_text(
        "cfg"
    )

    monkeypatch.setattr("tsqbev.openpcdet_env._find_executable", lambda name: None)
    monkeypatch.setattr(
        "tsqbev.openpcdet_env._module_version",
        lambda name: None if name in {"torch", "spconv", "torch_scatter"} else "0.0",
    )
    monkeypatch.setattr("tsqbev.openpcdet_env._read_torch_cuda_home", lambda: None)

    summary = check_openpcdet_environment(repo_root)

    assert summary["status"] == "blocked"
    assert "torch is not installed in the active Python environment" in summary["blockers"]
    assert "nvcc is not on PATH" in summary["blockers"]
    assert "spconv is not installed in the active Python environment" in summary["blockers"]
    assert "torch_scatter is not installed in the active Python environment" in summary["blockers"]


def test_check_openpcdet_environment_reports_ready(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "OpenPCDet"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "tools" / "cfgs" / "nuscenes_models").mkdir(parents=True)
    (repo_root / "docs" / "INSTALL.md").write_text("install")
    (repo_root / "tools" / "cfgs" / "nuscenes_models" / "cbgs_dyn_pp_centerpoint.yaml").write_text(
        "cfg"
    )

    def fake_find(name: str) -> str | None:
        if name == "nvcc":
            return "/usr/local/cuda/bin/nvcc"
        return None

    versions = {
        "torch": "2.1.0",
        "torchvision": "0.16.0",
        "spconv": "2.3.0",
        "torch_scatter": "2.1.2",
    }

    monkeypatch.setattr("tsqbev.openpcdet_env._find_executable", fake_find)
    monkeypatch.setattr("tsqbev.openpcdet_env._module_version", lambda name: versions.get(name))
    monkeypatch.setattr(
        "tsqbev.openpcdet_env._read_torch_cuda_home", lambda: "/usr/local/cuda"
    )

    summary = check_openpcdet_environment(repo_root)

    assert summary["status"] == "ready"
    assert summary["blockers"] == []
