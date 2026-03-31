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
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    bevfusion_root = args.bevfusion_root.resolve()
    dataset_root = args.dataset_root.resolve()
    sys.path.insert(0, str(bevfusion_root / "tools"))
    sys.path.insert(0, str(bevfusion_root))

    mmcv = importlib.import_module("mmcv")  # type: ignore
    NuScenes = importlib.import_module("nuscenes.nuscenes").NuScenes  # type: ignore[attr-defined]
    splits = importlib.import_module("nuscenes.utils.splits")  # type: ignore
    nuscenes_converter = importlib.import_module("tools.data_converter.nuscenes_converter")

    nusc = NuScenes(version=args.version, dataroot=str(dataset_root), verbose=True)
    available_scenes = nuscenes_converter.get_available_scenes(nusc)
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

    train_infos, val_infos = nuscenes_converter._fill_trainval_infos(  # type: ignore[attr-defined]
        nusc,
        train_tokens,
        val_tokens,
        False,
        max_sweeps=args.max_sweeps,
        max_radar_sweeps=args.max_sweeps,
    )

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
            "train_samples": len(train_infos),
            "val_samples": len(val_infos),
            "train_path": str(dataset_root / "nuscenes_infos_train.pkl"),
            "val_path": str(dataset_root / "nuscenes_infos_val.pkl"),
        }
    )


if __name__ == "__main__":
    main()
