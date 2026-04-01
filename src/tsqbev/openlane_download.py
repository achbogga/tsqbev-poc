"""Official OpenLane-V2 Google Drive download helpers.

References:
- OpenLane-V2 getting started:
  https://github.com/OpenDriveLab/OpenLane-V2/blob/main/docs/getting_started.md
- OpenLane-V2 data manifest:
  https://github.com/OpenDriveLab/OpenLane-V2/blob/main/data/README.md
"""

from __future__ import annotations

import hashlib
import tarfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OpenLaneV2Archive:
    key: str
    filename: str
    file_id: str
    md5: str
    description: str
    size_hint: str


_ARCHIVES: dict[str, OpenLaneV2Archive] = {
    "sample": OpenLaneV2Archive(
        key="sample",
        filename="OpenLane-V2_sample.tar",
        file_id="1Ni-L6u1MGKJRAfUXm39PdBIxdk_ntdc6",
        md5="21c607fa5a1930275b7f1409b25042a0",
        description="Public OpenLane-V2 sample tarball from the official README.",
        size_hint="~300M",
    ),
    "subset_a_info": OpenLaneV2Archive(
        key="subset_a_info",
        filename="OpenLane-V2_subset_A_info.tar",
        file_id="1t47lNF4H3WhSsAqgsl9lSLIeO0p6n8p4",
        md5="95bf28ccf22583d20434d75800be065d",
        description="Subset_A lane-centerline info archive.",
        size_hint="~8.8G",
    ),
    "subset_a_info_ls": OpenLaneV2Archive(
        key="subset_a_info_ls",
        filename="OpenLane-V2_subset_A_info-ls.tar.gz",
        file_id="14Wr2Gv2kyogY7_ZLEClqY0-Uhz4S109f",
        md5="1c1f9d49ecd47d6bc5bf093f38fb68c9",
        description="Subset_A map element bucket info archive.",
        size_hint="~240M",
    ),
    "subset_a_sdmap": OpenLaneV2Archive(
        key="subset_a_sdmap",
        filename="OpenLane-V2_subset_A_sdmap.tar",
        file_id="1nTsdxRZy_6N-itYndJujb-Ipwom6A5fh",
        md5="de22c7be880b667f1b3373ff665aac2e",
        description="Subset_A SDMap archive.",
        size_hint="~7M",
    ),
    "subset_b_info": OpenLaneV2Archive(
        key="subset_b_info",
        filename="OpenLane-V2_subset_B_info.tar",
        file_id="1Kn1tTwh9VrVa8nKwipL0bs9J5G9YLtIR",
        md5="27696b1ed1d99b1f70fdb68f439dc87d",
        description="Subset_B lane-centerline info archive.",
        size_hint="~7.7G",
    ),
}

_GOOGLE_WARNING_URL = "https://docs.google.com/uc?export=download&id={file_id}"
_GOOGLE_DOWNLOAD_URL = "https://drive.usercontent.google.com/download"
_DEFAULT_OUTPUT_DIR = Path("/home/achbogga/projects/research/openlanev2")


class _DriveFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "input":
            return
        payload = dict(attrs)
        name = payload.get("name")
        value = payload.get("value")
        if name and value is not None:
            self.inputs[name] = value


def openlanev2_archives() -> list[dict[str, str]]:
    return [asdict(archive) for archive in _ARCHIVES.values()]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_response(response: Any, output_path: Path) -> int:
    total = 0
    with output_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
    return total


def _stream_download(url: str, output_path: Path) -> tuple[int, str]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response:
        content_type = response.headers.get("Content-Type", "")
        bytes_written = _write_response(response, output_path)
    return bytes_written, content_type


def _resolve_google_drive_warning_params(
    file_id: str,
    scratch_html: Path,
) -> dict[str, str] | None:
    warning_url = _GOOGLE_WARNING_URL.format(file_id=file_id)
    _, content_type = _stream_download(warning_url, scratch_html)
    if "html" not in content_type.lower():
        return None
    parser = _DriveFormParser()
    parser.feed(scratch_html.read_text(errors="ignore"))
    params = {
        key: parser.inputs[key]
        for key in ("id", "export", "confirm", "uuid")
        if key in parser.inputs
    }
    if "id" not in params or "confirm" not in params:
        raise RuntimeError("failed to parse Google Drive warning form fields")
    return params


def _extract_archive(archive_path: Path, output_dir: Path) -> list[str]:
    extracted: list[str] = []
    with tarfile.open(archive_path, "r:*") as handle:
        members = handle.getmembers()
        handle.extractall(output_dir)
    for member in members[:20]:
        extracted.append(member.name)
    return extracted


def _detect_html_error(path: Path) -> str | None:
    prefix = path.read_text(errors="ignore")[:4096].lower()
    if "<html" not in prefix:
        return None
    if "quota exceeded" in prefix:
        return "google_drive_quota_exceeded"
    if "virus scan warning" in prefix:
        return "google_drive_virus_scan_warning"
    return "google_drive_html_response"


def download_openlanev2_archive(
    archive_key: str,
    *,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    extract: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    if archive_key not in _ARCHIVES:
        raise KeyError(f"unknown OpenLane-V2 archive key: {archive_key}")
    archive = _ARCHIVES[archive_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / archive.filename

    if archive_path.exists() and not overwrite:
        verified = _md5(archive_path) == archive.md5
        extracted_preview = (
            _extract_archive(archive_path, output_dir) if extract and verified else []
        )
        return {
            "status": "already_present",
            "archive": asdict(archive),
            "output_path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "md5_ok": verified,
            "extracted_preview": extracted_preview,
        }

    scratch_html = output_dir / f"{archive.filename}.warning.html"
    params = _resolve_google_drive_warning_params(archive.file_id, scratch_html)
    if params is None:
        scratch_html.replace(archive_path)
        bytes_written = archive_path.stat().st_size
    else:
        download_url = f"{_GOOGLE_DOWNLOAD_URL}?{urlencode(params)}"
        bytes_written, _ = _stream_download(download_url, archive_path)
        scratch_html.unlink(missing_ok=True)
    html_error = _detect_html_error(archive_path)
    if html_error is not None:
        error_path = output_dir / f"{archive.filename}.error.html"
        archive_path.replace(error_path)
        return {
            "status": "html_error",
            "archive": asdict(archive),
            "output_path": str(error_path),
            "bytes": bytes_written,
            "md5_ok": False,
            "error": html_error,
            "extracted_preview": [],
        }
    md5_ok = _md5(archive_path) == archive.md5
    extracted_preview = _extract_archive(archive_path, output_dir) if extract and md5_ok else []
    return {
        "status": "downloaded",
        "archive": asdict(archive),
        "output_path": str(archive_path),
        "bytes": bytes_written,
        "md5_ok": md5_ok,
        "extracted_preview": extracted_preview,
    }


def resolve_archive_keys(keys: Iterable[str] | None) -> list[str]:
    if keys is None:
        return ["sample"]
    resolved = [key.strip() for key in keys if key.strip()]
    if not resolved:
        return ["sample"]
    for key in resolved:
        if key not in _ARCHIVES:
            raise KeyError(f"unknown OpenLane-V2 archive key: {key}")
    return resolved
