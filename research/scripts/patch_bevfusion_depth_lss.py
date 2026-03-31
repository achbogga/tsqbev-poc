#!/usr/bin/env python3
"""Patch archived BEVFusion DepthLSSTransform to match its BaseDepthTransform contract."""

from __future__ import annotations

import argparse
from pathlib import Path

OLD_SNIPPET = """        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
"""

NEW_SNIPPET = """        dtransform_in_channels = 1 if self.depth_input == \"scalar\" else self.D
        if self.add_depth_features:
            dtransform_in_channels += 45 if self.use_points == \"radar\" else 5

        self.dtransform = nn.Sequential(
            nn.Conv2d(dtransform_in_channels, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bevfusion-root",
        type=Path,
        required=True,
        help="Path to the BEVFusion repo checkout to patch in-place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.bevfusion_root / "mmdet3d" / "models" / "vtransforms" / "depth_lss.py"
    source = target.read_text()

    if NEW_SNIPPET in source:
        print(target)
        print("already_patched")
        return

    if OLD_SNIPPET not in source:
        raise RuntimeError(f"unsupported depth_lss.py layout in {target}")

    target.write_text(source.replace(OLD_SNIPPET, NEW_SNIPPET))
    print(target)
    print("patched")


if __name__ == "__main__":
    main()
