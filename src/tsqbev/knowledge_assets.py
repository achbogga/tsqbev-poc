from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tsqbev.research_memory import REPO_ROOT

DEFAULT_RESEARCH_ASSET_ROOT = REPO_ROOT.parent / "research_assets"


def _sanitize_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)
    cleaned = cleaned.strip("-._")
    return cleaned or "asset"


def _repo_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


@dataclass(slots=True)
class KnowledgeAsset:
    asset_id: str
    collection: str
    knowledge_file: str
    entry_id: str
    title: str
    source_key: str
    kind: str
    url: str
    local_path: str
    status: str
    note: str | None = None
    source_hash: str | None = None


def _source_hash(url: str) -> str:
    import hashlib

    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _looks_like_remote_locator(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith(("http://", "https://", "git@"))


def _knowledge_json_files(repo_root: Path) -> list[Path]:
    return sorted(
        path
        for path in (repo_root / "research" / "knowledge").rglob("*.json")
        if path.is_file()
    )


def _asset_kind(source_key: str, url: str) -> str:
    lower_key = source_key.lower()
    lower_url = url.lower()
    if lower_url.endswith((".pth", ".pt", ".ckpt", ".safetensors", ".onnx")):
        return "checkpoint"
    if any(token in lower_key for token in ("checkpoint", "weight", "model")):
        return "checkpoint"
    if lower_url.endswith(".git") or "github.com" in lower_url:
        return "code"
    if (
        lower_url.endswith(".pdf")
        or "arxiv.org/abs/" in lower_url
        or "openaccess.thecvf.com" in lower_url
    ):
        return "paper"
    return "web"


def _asset_extension(kind: str, url: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if suffix:
        return suffix
    if kind == "paper":
        return ".pdf"
    if kind == "web":
        return ".html"
    if kind == "checkpoint":
        if "github.com" in url.lower():
            return ".html"
        return ".bin"
    return ""


def collect_knowledge_assets(
    repo_root: Path = REPO_ROOT,
    *,
    asset_root: Path = DEFAULT_RESEARCH_ASSET_ROOT,
) -> list[KnowledgeAsset]:
    assets: list[KnowledgeAsset] = []
    for path in _knowledge_json_files(repo_root):
        payload = json.loads(path.read_text())
        collection = str(payload.get("collection", path.stem))
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id", "entry"))
            title = str(entry.get("title", entry_id))
            sources = entry.get("sources", {})
            if not isinstance(sources, dict):
                continue
            for source_key, source_value in sources.items():
                if isinstance(source_value, str):
                    url = source_value.strip()
                    if not url or not _looks_like_remote_locator(url):
                        continue
                    kind = _asset_kind(str(source_key), url)
                    ext = _asset_extension(kind, url)
                    stem = _sanitize_component(str(source_key))
                    source_hash = _source_hash(url)
                    if kind == "code":
                        repo_name = Path(urlparse(url).path).name.removesuffix(".git")
                        target = (
                            asset_root
                            / "cache"
                            / "code"
                            / f"{source_hash}-{_sanitize_component(repo_name or stem)}"
                        )
                    else:
                        basename = Path(urlparse(url).path).name or f"{stem}{ext}"
                        target = (
                            asset_root
                            / "cache"
                            / kind
                            / f"{source_hash}-{_sanitize_component(basename)}"
                        )
                    assets.append(
                        KnowledgeAsset(
                            asset_id=f"{collection}:{entry_id}:{source_key}",
                            collection=collection,
                            knowledge_file=_repo_rel(path, repo_root),
                            entry_id=entry_id,
                            title=title,
                            source_key=str(source_key),
                            kind=kind,
                            url=url,
                            local_path=str(target),
                            status="planned",
                            source_hash=source_hash,
                        )
                    )
    return assets


def _mirror_repo(url: str, target: Path) -> tuple[str, str | None]:
    if (target / ".git").exists():
        return "present", "git repo already mirrored"
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return "mirrored", None
    return "error", (completed.stderr or completed.stdout or "git clone failed").strip()


def _mirror_file(url: str, target: Path) -> tuple[str, str | None]:
    if target.exists():
        return "present", "file already mirrored"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = Request(url, headers={"User-Agent": "tsqbev-knowledge-sync/1.0"})
        with urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
    except Exception as exc:
        return "error", repr(exc)
    return "mirrored", None


def mirror_knowledge_assets(
    repo_root: Path = REPO_ROOT,
    *,
    asset_root: Path = DEFAULT_RESEARCH_ASSET_ROOT,
    artifact_root: Path | None = None,
    max_workers: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    artifact_dir = artifact_root or (repo_root / "artifacts" / "knowledge_assets")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    assets = collect_knowledge_assets(repo_root, asset_root=asset_root)
    unique_assets: dict[str, KnowledgeAsset] = {}
    for asset in assets:
        unique_assets.setdefault(asset.url, asset)
    mirrored_lookup: dict[str, dict[str, Any]] = {}
    mirrored = 0
    present = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for asset in unique_assets.values():
            target = Path(asset.local_path)
            if overwrite and target.exists() and asset.kind != "code":
                target.unlink()
            elif overwrite and target.exists() and asset.kind == "code":
                shutil.rmtree(target)
            if asset.kind == "code":
                futures[executor.submit(_mirror_repo, asset.url, target)] = asset
            else:
                futures[executor.submit(_mirror_file, asset.url, target)] = asset
        for future in as_completed(futures):
            asset = futures[future]
            status, note = future.result()
            if status == "mirrored":
                mirrored += 1
            elif status == "present":
                present += 1
            else:
                errors += 1
            mirrored_lookup[asset.url] = {
                **asdict(asset),
                "status": status,
                "note": note,
            }
    results: list[KnowledgeAsset] = []
    for asset in assets:
        mirrored_asset = mirrored_lookup.get(
            asset.url,
            {**asdict(asset), "status": "error", "note": "missing mirrored result"},
        )
        results.append(KnowledgeAsset(**mirrored_asset))
    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "repo_root": str(repo_root),
        "asset_root": str(asset_root),
        "asset_count": len(results),
        "unique_asset_count": len(unique_assets),
        "max_workers": max_workers,
        "overwrite": overwrite,
        "mirrored_count": mirrored,
        "present_count": present,
        "error_count": errors,
        "collections": sorted({asset.collection for asset in results}),
        "assets": [asdict(item) for item in results],
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    (artifact_dir / "coverage_summary.json").write_text(
        json.dumps(
            {
                "generated_at_utc": summary["generated_at_utc"],
                "collections": summary["collections"],
                "asset_count": summary["asset_count"],
                "unique_asset_count": summary["unique_asset_count"],
                "mirrored_count": summary["mirrored_count"],
                "present_count": summary["present_count"],
                "error_count": summary["error_count"],
            },
            indent=2,
        )
    )
    return summary


def sync_knowledge_assets(
    repo_root: Path = REPO_ROOT,
    *,
    asset_root: Path = DEFAULT_RESEARCH_ASSET_ROOT,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    return mirror_knowledge_assets(
        repo_root=repo_root,
        asset_root=asset_root,
        artifact_root=artifact_root,
    )


def knowledge_asset_status(
    repo_root: Path = REPO_ROOT,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    artifact_dir = artifact_root or (repo_root / "artifacts" / "knowledge_assets")
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        return {
            "status": "missing",
            "artifact_root": str(artifact_dir),
        }
    return json.loads(manifest_path.read_text())
