#!/usr/bin/env python3
"""Patch archived BEVFusion nuScenes dataset runtime inconsistencies."""

from __future__ import annotations

import argparse
from pathlib import Path

PATCH_MARKER = "def _needs_radar_pipeline_input(self) -> bool:"

OLD_DATA_BLOCK = """        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info.get('location', None), 
            radar=info.get('radars', None), 
        )

        if data['location'] is None:
            data.pop('location')
        if data['radar'] is None:
            data.pop('radar')
"""

NEW_DATA_BLOCK = """        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info.get('location', None),
        )

        if data['location'] is None:
            data.pop('location')
        if self._needs_radar_pipeline_input():
            data['radar'] = info.get('radars', None)
            if data['radar'] is None:
                data.pop('radar')
"""

HELPER_INSERT_AFTER = """    def load_annotations(self, ann_file):
"""

HELPER_BLOCK = """    def _needs_radar_pipeline_input(self) -> bool:
        transforms = getattr(self.pipeline, "transforms", [])
        for transform in transforms:
            if transform.__class__.__name__ == "LoadRadarPointsMultiSweeps":
                return True
        return False

    def load_annotations(self, ann_file):
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bevfusion-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.bevfusion_root / "mmdet3d" / "datasets" / "nuscenes_dataset.py"
    source = target.read_text(encoding="utf-8")

    if PATCH_MARKER in source:
        print(target)
        print("already_patched")
        return

    if OLD_DATA_BLOCK not in source:
        raise RuntimeError("failed to locate archived NuScenesDataset data block to patch")
    if HELPER_INSERT_AFTER not in source:
        raise RuntimeError("failed to locate load_annotations insertion point")

    patched = source.replace(OLD_DATA_BLOCK, NEW_DATA_BLOCK)
    patched = patched.replace(HELPER_INSERT_AFTER, HELPER_BLOCK, 1)
    target.write_text(patched, encoding="utf-8")
    print(target)
    print("patched")


if __name__ == "__main__":
    main()
