from __future__ import annotations

import io
import tarfile
from pathlib import Path

from tsqbev.openlane_prepare import discover_openlane_v1_raw_layout, prepare_openlane_v1_from_raw


def _add_bytes_tar_entry(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _write_tar(path: Path, entries: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tf:
        for name, data in entries.items():
            _add_bytes_tar_entry(tf, name, data)


def test_prepare_openlane_v1_from_raw_extracts_trainer_layout(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    _write_tar(
        raw_root / "lane3d_300.tar",
        {
            "lane3d_300/training/segment-001/frame-001.json": (
                b'{"file_path": "training/segment-001/frame-001.jpg",'
                b' "intrinsic": [[1,0,0],[0,1,0],[0,0,1]],'
                b' "extrinsic": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],'
                b' "lane_lines": []}'
            ),
            "lane3d_300/validation/segment-002/frame-002.json": (
                b'{"file_path": "validation/segment-002/frame-002.jpg",'
                b' "intrinsic": [[1,0,0],[0,1,0],[0,0,1]],'
                b' "extrinsic": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],'
                b' "lane_lines": []}'
            ),
        },
    )
    _write_tar(
        raw_root / "images_training_0.tar",
        {"images_training_0/segment-001/frame-001.jpg": b"train-image"},
    )
    _write_tar(
        raw_root / "images_validation_0.tar",
        {"images_validation_0/segment-002/frame-002.jpg": b"val-image"},
    )

    layout = discover_openlane_v1_raw_layout(tmp_path)
    assert layout.lane3d_300_tar.name == "lane3d_300.tar"
    assert [path.name for path in layout.training_image_tars] == ["images_training_0.tar"]
    assert [path.name for path in layout.validation_image_tars] == ["images_validation_0.tar"]

    report = prepare_openlane_v1_from_raw(tmp_path)
    assert report["ready"] is True
    assert (tmp_path / "lane3d_300" / "training" / "segment-001" / "frame-001.json").exists()
    assert (tmp_path / "images" / "training" / "segment-001" / "frame-001.jpg").exists()
    assert (tmp_path / "images" / "validation" / "segment-002" / "frame-002.jpg").exists()


def test_prepare_openlane_v1_from_raw_is_resumable(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_tar(
        raw_root / "lane3d_300.tar",
        {"lane3d_300/training/segment-001/frame-001.json": b"{}"},
    )
    _write_tar(
        raw_root / "images_training_0.tar",
        {"images_training_0/segment-001/frame-001.jpg": b"train-image"},
    )
    _write_tar(
        raw_root / "images_validation_0.tar",
        {"images_validation_0/segment-002/frame-002.jpg": b"val-image"},
    )
    (tmp_path / "lane3d_300" / "validation").mkdir(parents=True)

    first = prepare_openlane_v1_from_raw(tmp_path)
    second = prepare_openlane_v1_from_raw(tmp_path)

    assert any(action["status"] == "extracted" for action in first["actions"])
    assert all(action["status"] == "skipped" for action in second["actions"])
