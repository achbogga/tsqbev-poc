#!/usr/bin/env python3
"""Generate BEVFusion-compatible nuScenes ann files without GT DB creation.

Primary sources:
- BEVFusion README:
  https://github.com/mit-han-lab/bevfusion
- BEVFusion nuScenes converter:
  https://github.com/mit-han-lab/bevfusion/blob/main/tools/data_converter/nuscenes_converter.py
- Archived BEVFusion issue documenting create_data.py failure on mini:
  https://github.com/mit-han-lab/bevfusion/issues/569
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare BEVFusion-compatible nuScenes ann files without GT DB creation."
    )
    parser.add_argument("--bevfusion-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--version",
        choices=("v1.0-trainval", "v1.0-mini"),
        default="v1.0-trainval",
    )
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=("eval-only", "trainval", "location-only"),
        default="eval-only",
        help=(
            "Use eval-only to generate the val ann file needed by tools/test.py "
            "without traversing every train sample. Use location-only to patch "
            "existing info files in-place with map metadata needed by segmentation."
        ),
    )
    return parser


def _load_name_mapping(bevfusion_root: Path) -> dict[str, str]:
    dataset_path = bevfusion_root / "mmdet3d" / "datasets" / "nuscenes_dataset.py"
    module = ast.parse(dataset_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "NuScenesDataset":
            for class_node in node.body:
                if not isinstance(class_node, ast.Assign):
                    continue
                for target in class_node.targets:
                    if isinstance(target, ast.Name) and target.id == "NameMapping":
                        return ast.literal_eval(class_node.value)
    raise RuntimeError(
        "failed to locate NuScenesDataset.NameMapping in upstream nuscenes_dataset.py"
    )


def _load_converter_namespace(bevfusion_root: Path) -> dict[str, object]:
    converter_path = bevfusion_root / "tools" / "data_converter" / "nuscenes_converter.py"
    source = converter_path.read_text(encoding="utf-8")
    source = source.replace("from mmdet3d.core.bbox.box_np_ops import points_cam2img\n", "")
    source = source.replace("from mmdet3d.datasets import NuScenesDataset\n", "")
    source = source.replace(
        """            locs = np.array([b.center for b in boxes]).reshape(-1, 3)\n"""
        """            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)\n"""
        """            rots = np.array([b.orientation.yaw_pitch_roll[0]\n"""
        """                             for b in boxes]).reshape(-1, 1)\n""",
        """            locs = np.array([anno['translation'] for anno in annotations])"""
        """.reshape(-1, 3)\n"""
        """            dims = np.array([anno['size'] for anno in annotations])"""
        """.reshape(-1, 3)\n"""
        """            rots = np.array([\n"""
        """                Quaternion(anno['rotation']).yaw_pitch_roll[0]\n"""
        """                for anno in annotations\n"""
        """            ]).reshape(-1, 1)\n""",
    )
    source = source.replace(
        """            names = [b.name for b in boxes]\n""",
        """            names = [anno['category_name'] for anno in annotations]\n""",
    )
    source = source.replace(
        """            for i in range(len(boxes)):\n""",
        """            for i in range(len(annotations)):\n""",
    )
    source = source.replace(
        """            assert len(gt_boxes) == len(\n"""
        """                annotations), f'{len(gt_boxes)}, {len(annotations)}'\n""",
        "",
    )

    def _points_cam2img_unused(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "points_cam2img should not be reached when generating nuscenes_infos_train/val.pkl"
        )

    class _NuScenesDatasetShim:
        NameMapping = _load_name_mapping(bevfusion_root)

    namespace: dict[str, object] = {
        "__file__": str(converter_path),
        "__name__": "tsqbev_prepare_nuscenes_converter",
        "points_cam2img": _points_cam2img_unused,
        "NuScenesDataset": _NuScenesDatasetShim,
    }
    exec(compile(source, str(converter_path), "exec"), namespace)
    return namespace


def _sample_location_mapping(nusc: object) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        log = nusc.get("log", scene["log_token"])
        mapping[sample["token"]] = log["location"]
    return mapping


def _inject_locations(infos: list[dict[str, object]], location_by_token: dict[str, str]) -> None:
    for info in infos:
        token = info["token"]
        if not isinstance(token, str):
            raise RuntimeError(f"unexpected non-string sample token in info: {token!r}")
        location = location_by_token[token]
        info["location"] = location
        info["map_location"] = location


def _patch_existing_info_file(path: Path, location_by_token: dict[str, str], mmcv: object) -> int:
    if not path.exists():
        return 0
    payload = mmcv.load(path)
    infos = payload.get("infos", [])
    _inject_locations(infos, location_by_token)
    mmcv.dump(payload, path)
    return len(infos)


def main() -> None:
    args = _build_parser().parse_args()

    bevfusion_root = args.bevfusion_root.resolve()
    dataset_root = args.dataset_root.resolve()
    sys.path.insert(0, str(bevfusion_root / "tools"))
    sys.path.insert(0, str(bevfusion_root))

    mmcv = importlib.import_module("mmcv")  # type: ignore
    NuScenes = importlib.import_module("nuscenes.nuscenes").NuScenes  # type: ignore[attr-defined]
    splits = importlib.import_module("nuscenes.utils.splits")  # type: ignore
    nuscenes_converter = _load_converter_namespace(bevfusion_root)

    nusc = NuScenes(version=args.version, dataroot=str(dataset_root), verbose=True)
    location_by_token = _sample_location_mapping(nusc)
    available_scenes = nuscenes_converter["get_available_scenes"](nusc)
    available_scene_names = [scene["name"] for scene in available_scenes]

    if args.version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    else:
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val

    train_tokens = {
        available_scenes[available_scene_names.index(name)]["token"]
        for name in train_scenes
        if name in available_scene_names
    }
    val_tokens = {
        available_scenes[available_scene_names.index(name)]["token"]
        for name in val_scenes
        if name in available_scene_names
    }

    if args.mode == "location-only":
        train_count = _patch_existing_info_file(
            dataset_root / "nuscenes_infos_train.pkl",
            location_by_token,
            mmcv,
        )
        val_count = _patch_existing_info_file(
            dataset_root / "nuscenes_infos_val.pkl",
            location_by_token,
            mmcv,
        )
        print(
            {
                "dataset_root": str(dataset_root),
                "version": args.version,
                "mode": args.mode,
                "train_samples": train_count,
                "val_samples": val_count,
                "train_path": str(dataset_root / "nuscenes_infos_train.pkl"),
                "val_path": str(dataset_root / "nuscenes_infos_val.pkl"),
            }
        )
        return

    if args.mode == "eval-only":
        filtered_samples = [sample for sample in nusc.sample if sample["scene_token"] in val_tokens]
        original_track_iter_progress = mmcv.track_iter_progress

        def _filtered_track_iter_progress(iterable: object):
            if iterable is nusc.sample:
                return original_track_iter_progress(filtered_samples)
            return original_track_iter_progress(iterable)

        try:
            mmcv.track_iter_progress = _filtered_track_iter_progress
            train_infos, val_infos = nuscenes_converter["_fill_trainval_infos"](
                nusc,
                set(),
                val_tokens,
                False,
                max_sweeps=args.max_sweeps,
                max_radar_sweeps=args.max_sweeps,
            )
        finally:
            mmcv.track_iter_progress = original_track_iter_progress
    else:
        train_infos, val_infos = nuscenes_converter["_fill_trainval_infos"](
            nusc,
            train_tokens,
            val_tokens,
            False,
            max_sweeps=args.max_sweeps,
            max_radar_sweeps=args.max_sweeps,
        )

    _inject_locations(train_infos, location_by_token)
    _inject_locations(val_infos, location_by_token)

    metadata = {"version": args.version}
    mmcv.dump(
        {"infos": train_infos, "metadata": metadata},
        dataset_root / "nuscenes_infos_train.pkl",
    )
    mmcv.dump(
        {"infos": val_infos, "metadata": metadata},
        dataset_root / "nuscenes_infos_val.pkl",
    )

    print(
        {
            "dataset_root": str(dataset_root),
            "version": args.version,
            "mode": args.mode,
            "train_samples": len(train_infos),
            "val_samples": len(val_infos),
            "train_path": str(dataset_root / "nuscenes_infos_train.pkl"),
            "val_path": str(dataset_root / "nuscenes_infos_val.pkl"),
        }
    )


if __name__ == "__main__":
    main()
