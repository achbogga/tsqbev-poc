"""Prepare OpenLane V1 raw archives into the trainer-ready layout.

References:
- OpenLane official dataset layout and download notes:
  https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tsqbev.data_checks import check_openlane_root


@dataclass(frozen=True)
class OpenLaneV1RawLayout:
    root: Path
    raw_root: Path
    lane3d_300_tar: Path
    training_image_tars: tuple[Path, ...]
    validation_image_tars: tuple[Path, ...]
    lane3d_1000_zip: Path | None
    scene_zip: Path | None
    cipo_zip: Path | None


def discover_openlane_v1_raw_layout(dataroot: str | Path) -> OpenLaneV1RawLayout:
    root = Path(dataroot)
    raw_root = root / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"expected raw archive directory at {raw_root}")

    lane3d_300_tar = raw_root / "lane3d_300.tar"
    if not lane3d_300_tar.exists():
        raise FileNotFoundError(f"missing {lane3d_300_tar}")

    training_image_tars = tuple(sorted(raw_root.glob("images_training_*.tar")))
    validation_image_tars = tuple(sorted(raw_root.glob("images_validation_*.tar")))
    if not training_image_tars:
        raise FileNotFoundError(f"no training image shards found under {raw_root}")
    if not validation_image_tars:
        raise FileNotFoundError(f"no validation image shards found under {raw_root}")

    lane3d_1000_zip = raw_root / "lane3d_1000_v1.2.zip"
    scene_zip = raw_root / "scene.zip"
    cipo_zip = raw_root / "cipo.zip"

    return OpenLaneV1RawLayout(
        root=root,
        raw_root=raw_root,
        lane3d_300_tar=lane3d_300_tar,
        training_image_tars=training_image_tars,
        validation_image_tars=validation_image_tars,
        lane3d_1000_zip=lane3d_1000_zip if lane3d_1000_zip.exists() else None,
        scene_zip=scene_zip if scene_zip.exists() else None,
        cipo_zip=cipo_zip if cipo_zip.exists() else None,
    )


def _state_path(root: Path) -> Path:
    return root / ".tsqbev_openlane_v1_prepare_state.json"


def _load_state(root: Path) -> dict[str, object]:
    path = _state_path(root)
    if not path.exists():
        return {"completed": []}
    return json.loads(path.read_text())


def _save_state(root: Path, state: dict[str, object]) -> None:
    _state_path(root).write_text(json.dumps(state, indent=2, sort_keys=True))


def _mark_completed(root: Path, state: dict[str, object], key: str) -> None:
    existing = state.get("completed", [])
    completed = list(existing) if isinstance(existing, list) else []
    if key not in completed:
        completed.append(key)
        state["completed"] = completed
        _save_state(root, state)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _extract_tar(archive: Path, destination: Path, *, strip_components: int = 0) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "tar",
        "-xf",
        str(archive),
        "-C",
        str(destination),
        "--skip-old-files",
    ]
    if strip_components:
        command.extend(["--strip-components", str(strip_components)])
    _run(command)


def _extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    _run(["unzip", "-n", str(archive), "-d", str(destination)])


def prepare_openlane_v1_from_raw(
    dataroot: str | Path,
    *,
    include_lane3d_1000: bool = False,
    include_scene: bool = False,
    include_cipo: bool = False,
    force_reextract: bool = False,
) -> dict[str, object]:
    layout = discover_openlane_v1_raw_layout(dataroot)
    state = _load_state(layout.root)
    if force_reextract:
        state = {"completed": []}

    completed_raw = state.get("completed", [])
    completed = set(completed_raw) if isinstance(completed_raw, list) else set()
    actions: list[dict[str, object]] = []

    lane_key = "lane3d_300"
    if lane_key not in completed:
        print(f"[prepare-openlanev1] extracting {layout.lane3d_300_tar.name}", flush=True)
        _extract_tar(layout.lane3d_300_tar, layout.root)
        _mark_completed(layout.root, state, lane_key)
        actions.append({"archive": layout.lane3d_300_tar.name, "status": "extracted"})
    else:
        actions.append({"archive": layout.lane3d_300_tar.name, "status": "skipped"})

    training_root = layout.root / "images" / "training"
    validation_root = layout.root / "images" / "validation"
    for archive in layout.training_image_tars:
        key = f"training::{archive.name}"
        if key not in completed:
            print(f"[prepare-openlanev1] extracting {archive.name} -> images/training", flush=True)
            _extract_tar(archive, training_root, strip_components=1)
            _mark_completed(layout.root, state, key)
            actions.append({"archive": archive.name, "status": "extracted"})
        else:
            actions.append({"archive": archive.name, "status": "skipped"})
    for archive in layout.validation_image_tars:
        key = f"validation::{archive.name}"
        if key not in completed:
            print(
                f"[prepare-openlanev1] extracting {archive.name} -> images/validation",
                flush=True,
            )
            _extract_tar(archive, validation_root, strip_components=1)
            _mark_completed(layout.root, state, key)
            actions.append({"archive": archive.name, "status": "extracted"})
        else:
            actions.append({"archive": archive.name, "status": "skipped"})

    if include_lane3d_1000 and layout.lane3d_1000_zip is not None:
        key = "lane3d_1000"
        if key not in completed:
            print(f"[prepare-openlanev1] extracting {layout.lane3d_1000_zip.name}", flush=True)
            _extract_zip(layout.lane3d_1000_zip, layout.root)
            _mark_completed(layout.root, state, key)
            actions.append({"archive": layout.lane3d_1000_zip.name, "status": "extracted"})
        else:
            actions.append({"archive": layout.lane3d_1000_zip.name, "status": "skipped"})

    if include_scene and layout.scene_zip is not None:
        key = "scene"
        if key not in completed:
            print(f"[prepare-openlanev1] extracting {layout.scene_zip.name}", flush=True)
            _extract_zip(layout.scene_zip, layout.root)
            _mark_completed(layout.root, state, key)
            actions.append({"archive": layout.scene_zip.name, "status": "extracted"})
        else:
            actions.append({"archive": layout.scene_zip.name, "status": "skipped"})

    if include_cipo and layout.cipo_zip is not None:
        key = "cipo"
        if key not in completed:
            print(f"[prepare-openlanev1] extracting {layout.cipo_zip.name}", flush=True)
            _extract_zip(layout.cipo_zip, layout.root)
            _mark_completed(layout.root, state, key)
            actions.append({"archive": layout.cipo_zip.name, "status": "extracted"})
        else:
            actions.append({"archive": layout.cipo_zip.name, "status": "skipped"})

    readiness = check_openlane_root(layout.root)
    return {
        "dataset": "OpenLane",
        "root": str(layout.root),
        "raw_root": str(layout.raw_root),
        "ready": readiness["ready"],
        "checks": readiness["checks"],
        "actions": actions,
        "state_path": str(_state_path(layout.root)),
    }
