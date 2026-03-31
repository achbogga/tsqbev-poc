#!/usr/bin/env python3
"""Patch archived BEVFusion checkpoints for local compatibility fixes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

DEPTH_STEM_KEY = "encoders.camera.vtransform.dtransform.0.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Original checkpoint path.")
    parser.add_argument("--output", type=Path, required=True, help="Patched checkpoint path.")
    parser.add_argument(
        "--target-depth-channels",
        type=int,
        default=6,
        help="Expected input channels for the patched depth stem.",
    )
    return parser.parse_args()


def _patch_state_dict(state_dict: dict[str, torch.Tensor], target_depth_channels: int) -> bool:
    weight = state_dict.get(DEPTH_STEM_KEY)
    if weight is None:
        return False

    if weight.ndim != 4 or weight.shape[1] == target_depth_channels:
        return False

    if weight.shape[1] != 1:
        raise RuntimeError(
            f"unexpected depth stem shape {tuple(weight.shape)} for {DEPTH_STEM_KEY}; "
            "only scalar-depth checkpoints are supported"
        )

    patched = torch.zeros(
        (weight.shape[0], target_depth_channels, weight.shape[2], weight.shape[3]),
        dtype=weight.dtype,
    )
    patched[:, :1, :, :] = weight
    state_dict[DEPTH_STEM_KEY] = patched
    return True


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"unsupported checkpoint format in {args.input}")

    patched = _patch_state_dict(state_dict, args.target_depth_channels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(args.output)
    print("patched" if patched else "unchanged")


if __name__ == "__main__":
    main()
