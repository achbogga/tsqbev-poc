#!/usr/bin/env python3
"""Build the missing BEVFusion feature_decorator CUDA extension in place.

Primary sources:
- Official BEVFusion repo:
  https://github.com/mit-han-lab/bevfusion
- Upstream setup.py, which omits this extension from ext_modules:
  https://github.com/mit-han-lab/bevfusion/blob/main/setup.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from torch.utils.cpp_extension import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bevfusion-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bevfusion_root = args.bevfusion_root.resolve()
    package_dir = bevfusion_root / "mmdet3d" / "ops" / "feature_decorator"
    source_dir = package_dir / "src"

    if not package_dir.exists():
        raise FileNotFoundError(f"missing feature_decorator package at {package_dir}")

    existing = next(package_dir.glob("feature_decorator_ext*.so"), None)
    if existing is not None and not args.force:
        print(existing)
        return

    build_dir = bevfusion_root / "build" / "feature_decorator_ext"
    build_dir.mkdir(parents=True, exist_ok=True)

    compat_cpp = build_dir / "feature_decorator_compat.cpp"
    compat_cpp.write_text(
        (source_dir / "feature_decorator.cpp")
        .read_text(encoding="utf-8")
        .replace(
            "  int normalize_coords, int use_cluster, int use_center\n",
            "  int64_t normalize_coords, int64_t use_cluster, int64_t use_center\n",
        ),
        encoding="utf-8",
    )

    module = load(
        name="feature_decorator_ext",
        sources=[
            str(compat_cpp),
            str(source_dir / "feature_decorator_cuda.cu"),
        ],
        build_directory=str(build_dir),
        verbose=True,
        extra_cuda_cflags=[
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
        ],
    )

    built_extension = Path(module.__file__).resolve()
    destination = package_dir / built_extension.name
    shutil.copy2(built_extension, destination)
    print(destination)


if __name__ == "__main__":
    main()
