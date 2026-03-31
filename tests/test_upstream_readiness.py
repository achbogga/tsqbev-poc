from __future__ import annotations

from pathlib import Path

from tsqbev.upstream_readiness import check_upstream_stack


def _touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_upstream_readiness_reports_present_repo(tmp_path: Path) -> None:
    repo_root = tmp_path / "OpenPCDet_official"
    _touch(repo_root, "pcdet/__init__.py")
    _touch(repo_root, "tools/cfgs/.keep")
    _touch(repo_root, "requirements.txt")

    statuses = {status.key: status for status in check_upstream_stack(tmp_path)}

    assert statuses["openpcdet"].present is True
    assert statuses["openpcdet"].missing_files == ()


def test_upstream_readiness_reports_missing_repo_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "BEVDet"
    _touch(repo_root, "configs/.keep")

    statuses = {status.key: status for status in check_upstream_stack(tmp_path)}

    assert statuses["bevdet"].present is False
    assert "mmdet3d" in statuses["bevdet"].missing_files
