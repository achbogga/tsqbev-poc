from __future__ import annotations

import json
from pathlib import Path

from tsqbev.knowledge_assets import (
    _knowledge_json_files,
    _source_hash,
    collect_knowledge_assets,
    knowledge_asset_status,
    mirror_knowledge_assets,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_repo_fixture(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    _write(
        repo_root / "research" / "knowledge" / "nested" / "sample_pack.json",
        json.dumps(
            {
                "collection": "sample_pack",
                "entries": [
                    {
                        "id": "sam2",
                        "title": "SAM 2.1",
                        "sources": {
                            "paper": "https://example.com/paper.pdf",
                            "code": "https://github.com/example/sam2",
                        },
                    },
                    {
                        "id": "sam2_alias",
                        "title": "SAM 2.1 Alias",
                        "sources": {
                            "paper": "https://example.com/paper.pdf"
                        },
                    },
                ],
            },
            indent=2,
        ),
    )
    return repo_root


def test_knowledge_json_files_recurse(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)

    files = _knowledge_json_files(repo_root)

    assert len(files) == 1
    assert files[0].name == "sample_pack.json"


def test_collect_knowledge_assets_uses_deduped_cache_paths(tmp_path: Path) -> None:
    repo_root = _make_repo_fixture(tmp_path)
    asset_root = tmp_path / "assets"

    assets = collect_knowledge_assets(repo_root, asset_root=asset_root)

    assert len(assets) == 3
    paper_assets = [asset for asset in assets if asset.source_key == "paper"]
    assert len(paper_assets) == 2
    assert paper_assets[0].local_path == paper_assets[1].local_path
    assert paper_assets[0].source_hash == _source_hash("https://example.com/paper.pdf")
    assert all(asset.local_path.startswith(str(asset_root / "cache")) for asset in assets)


def test_mirror_knowledge_assets_writes_deduped_manifest(monkeypatch, tmp_path: Path) -> None:
    import tsqbev.knowledge_assets as knowledge_assets

    repo_root = _make_repo_fixture(tmp_path)
    asset_root = tmp_path / "assets"
    artifact_root = repo_root / "artifacts" / "knowledge_assets"

    def _fake_repo(url: str, target: Path):  # type: ignore[no-untyped-def]
        target.mkdir(parents=True, exist_ok=True)
        return "mirrored", None

    def _fake_file(url: str, target: Path):  # type: ignore[no-untyped-def]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("payload")
        return "mirrored", None

    monkeypatch.setattr(knowledge_assets, "_mirror_repo", _fake_repo)
    monkeypatch.setattr(knowledge_assets, "_mirror_file", _fake_file)

    summary = mirror_knowledge_assets(
        repo_root=repo_root,
        asset_root=asset_root,
        artifact_root=artifact_root,
        max_workers=2,
    )
    status = knowledge_asset_status(repo_root=repo_root, artifact_root=artifact_root)

    assert summary["asset_count"] == 3
    assert summary["unique_asset_count"] == 2
    assert summary["error_count"] == 0
    assert (artifact_root / "manifest.json").exists()
    assert status["unique_asset_count"] == 2
    assert len(status["assets"]) == 3
