#!/usr/bin/env python3
"""Prepare BEVDet-style nuScenes info PKLs with parameterized roots and versions."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

MAP_NAME_FROM_GENERAL_TO_DETECTION = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}
CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


def _get_gt(info: dict) -> tuple[list[np.ndarray], list[int]]:
    ego2global_rotation = info["cams"]["CAM_FRONT"]["ego2global_rotation"]
    ego2global_translation = info["cams"]["CAM_FRONT"]["ego2global_translation"]
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes: list[np.ndarray] = []
    gt_labels: list[int] = []
    for ann_info in info["ann_infos"]:
        mapped = MAP_NAME_FROM_GENERAL_TO_DETECTION[ann_info["category_name"]]
        if mapped not in CLASSES or ann_info["num_lidar_pts"] + ann_info["num_radar_pts"] <= 0:
            continue
        box = Box(
            ann_info["translation"],
            ann_info["size"],
            Quaternion(ann_info["rotation"]),
            velocity=ann_info["velocity"],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(CLASSES.index(mapped))
    return gt_boxes, gt_labels


def _add_ann_adj_info(dataset_root: Path, version: str, extra_tag: str) -> dict[str, int]:
    nuscenes = NuScenes(version=version, dataroot=str(dataset_root))
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        path = dataset_root / f"{extra_tag}_infos_{split}.pkl"
        with path.open("rb") as handle:
            dataset = pickle.load(handle)
        infos = dataset["infos"]
        counts[split] = len(infos)
        for index, info in enumerate(infos):
            if index % 50 == 0:
                print(f"[prepare-bevdet] {split} {index}/{len(infos)}")
            sample = nuscenes.get("sample", info["token"])
            ann_infos: list[dict] = []
            for ann_token in sample["anns"]:
                ann_info = nuscenes.get("sample_annotation", ann_token)
                velocity = nuscenes.box_velocity(ann_info["token"])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info["velocity"] = velocity
                ann_infos.append(ann_info)
            info["ann_infos"] = _get_gt({"cams": info["cams"], "ann_infos": ann_infos})
            info["scene_token"] = sample["scene_token"]
            scene = nuscenes.get("scene", sample["scene_token"])
            info["occ_path"] = f"{dataset_root}/gts/{scene['name']}/{info['token']}"
        with path.open("wb") as handle:
            pickle.dump(dataset, handle)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bevdet-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--extra-tag", type=str, required=True)
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.bevdet_root))
    from tools.data_converter import nuscenes_converter  # noqa: PLC0415

    train_path = args.dataset_root / f"{args.extra_tag}_infos_train.pkl"
    val_path = args.dataset_root / f"{args.extra_tag}_infos_val.pkl"
    if not args.overwrite and train_path.exists() and val_path.exists():
        print(
            {
                "status": "already_present",
                "train_info": str(train_path),
                "val_info": str(val_path),
                "version": args.version,
            }
        )
        return

    nuscenes_converter.create_nuscenes_infos(
        str(args.dataset_root),
        args.extra_tag,
        version=args.version,
        max_sweeps=args.max_sweeps,
    )
    counts = _add_ann_adj_info(args.dataset_root, args.version, args.extra_tag)
    print(
        {
            "status": "prepared",
            "dataset_root": str(args.dataset_root),
            "version": args.version,
            "extra_tag": args.extra_tag,
            "train_info": str(train_path),
            "val_info": str(val_path),
            "counts": counts,
        }
    )


if __name__ == "__main__":
    main()
