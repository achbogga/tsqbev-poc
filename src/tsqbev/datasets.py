"""Real public dataset loaders and minimal batch collation.

References:
- nuScenes official devkit:
  https://github.com/nutonomy/nuscenes-devkit
- OpenLane official dataset and evaluation docs:
  https://github.com/OpenDriveLab/OpenLane
- OpenLane 3D evaluation transform logic:
  https://github.com/OpenDriveLab/OpenLane/blob/main/eval/LANE_evaluation/lane3d/eval_3D_lane.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from tsqbev.contracts import LaneTargets, ObjectTargets, SceneBatch
from tsqbev.labels import (
    NUSCENES_CAMERA_NAMES,
    NUSCENES_DETECTION_NAME_TO_INDEX,
)
from tsqbev.quaternion import rotate_xy, transform_from_quaternion, wrap_angle, yaw_from_quaternion

Tensor = torch.Tensor

_OPENLANE_R_VG = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
_OPENLANE_R_GC = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
_OPENLANE_CAM_REPRESENTATION = np.linalg.inv(
    np.array(
        [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
)


@dataclass(slots=True)
class SceneExample:
    """One dataset example with the canonical scene batch and export metadata."""

    scene: SceneBatch
    metadata: dict[str, Any]


def collate_single_scene_example(
    examples: list[SceneExample],
) -> tuple[SceneBatch, list[dict[str, Any]]]:
    """Collate a single-example batch.

    The first real baseline is intentionally local and memory-bounded on a 16 GB RTX 5000,
    so the public training loop uses batch size 1 with gradient accumulation.
    """

    if len(examples) != 1:
        raise ValueError("real-dataset collation currently supports batch_size=1 only")
    scene = examples[0].scene
    scene.validate()
    return scene, [examples[0].metadata]


def _image_to_tensor(path: Path, image_size: tuple[int, int]) -> tuple[Tensor, tuple[int, int]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        original_size = rgb.height, rgb.width
        resized = rgb.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
        tensor = pil_to_tensor(resized).float() / 255.0
    return tensor, original_size


def _scale_intrinsics(
    intrinsics: np.ndarray, original_size_hw: tuple[int, int], target_size_hw: tuple[int, int]
) -> np.ndarray:
    scaled = intrinsics.astype(np.float32).copy()
    scale_y = target_size_hw[0] / float(original_size_hw[0])
    scale_x = target_size_hw[1] / float(original_size_hw[1])
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def _load_nuscenes_devkit() -> tuple[Any, Any, Any, Any]:
    try:
        from nuscenes.eval.detection.utils import category_to_detection_name
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as exc:  # pragma: no cover - exercised by real runs, not unit tests.
        raise RuntimeError(
            "nuScenes support requires `uv sync --extra data` to install nuscenes-devkit"
        ) from exc
    return NuScenes, LidarPointCloud, category_to_detection_name, create_splits_scenes


def _resample_polyline_in_y(points_xyz: np.ndarray, num_points: int) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    order = np.argsort(points[:, 1], kind="stable")
    points = points[order]
    unique_y, unique_indices = np.unique(points[:, 1], return_index=True)
    points = points[unique_indices]
    if points.shape[0] == 1:
        return np.repeat(points, repeats=num_points, axis=0)
    sample_y = np.linspace(float(unique_y[0]), float(unique_y[-1]), num=num_points, endpoint=True)
    x = np.interp(sample_y, points[:, 1], points[:, 0])
    z = np.interp(sample_y, points[:, 1], points[:, 2])
    return np.stack((x, sample_y, z), axis=-1).astype(np.float32)


def _openlane_camera_to_ground_extrinsic(original_extrinsic: np.ndarray) -> np.ndarray:
    """Match the official OpenLane 3D evaluation transform setup."""

    camera_to_ground = original_extrinsic.astype(np.float32).copy()
    camera_to_ground[:3, :3] = (
        np.linalg.inv(_OPENLANE_R_VG)
        @ camera_to_ground[:3, :3]
        @ _OPENLANE_R_VG
        @ _OPENLANE_R_GC
    )
    camera_to_ground[0:2, 3] = 0.0
    return camera_to_ground


def _openlane_lane_points_to_ground(
    original_extrinsic: np.ndarray, lane_xyz_camera_waymo: np.ndarray
) -> np.ndarray:
    camera_to_ground = _openlane_camera_to_ground_extrinsic(original_extrinsic)
    lane_h = np.concatenate(
        (
            lane_xyz_camera_waymo.astype(np.float32),
            np.ones((lane_xyz_camera_waymo.shape[0], 1), dtype=np.float32),
        ),
        axis=-1,
    )
    transformed = lane_h @ _OPENLANE_CAM_REPRESENTATION.T @ camera_to_ground.T
    return transformed[:, :3]


class NuScenesDataset(Dataset[SceneExample]):
    """nuScenes train/val dataset in the canonical TSQBEV batch contract."""

    def __init__(
        self,
        dataroot: str | Path,
        version: str = "v1.0-trainval",
        split: str = "train",
        image_size: tuple[int, int] = (256, 704),
        verbose: bool = False,
    ) -> None:
        super().__init__()
        NuScenes, _, _, create_splits_scenes = _load_nuscenes_devkit()
        self.dataroot = Path(dataroot)
        self.version = version
        self.split = split
        self.image_size = image_size
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=verbose)

        scene_names = set(create_splits_scenes(verbose=False)[split])
        scene_token_to_name = {scene["token"]: scene["name"] for scene in self.nusc.scene}
        split_samples = [
            sample
            for sample in self.nusc.sample
            if scene_token_to_name[sample["scene_token"]] in scene_names
        ]
        ordered_samples = sorted(split_samples, key=lambda item: item["timestamp"])
        self.sample_tokens = [sample["token"] for sample in ordered_samples]

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, index: int) -> SceneExample:
        _, LidarPointCloud, category_to_detection_name, _ = _load_nuscenes_devkit()

        sample_token = self.sample_tokens[index]
        sample = self.nusc.get("sample", sample_token)

        image_tensors: list[Tensor] = []
        intrinsics: list[np.ndarray] = []
        extrinsics: list[np.ndarray] = []

        for camera_name in NUSCENES_CAMERA_NAMES:
            sample_data = self.nusc.get("sample_data", sample["data"][camera_name])
            calibrated_sensor = self.nusc.get(
                "calibrated_sensor", sample_data["calibrated_sensor_token"]
            )
            image_path = self.dataroot / sample_data["filename"]
            image_tensor, original_size = _image_to_tensor(image_path, self.image_size)
            image_tensors.append(image_tensor)

            intrinsic = np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float32)
            intrinsics.append(_scale_intrinsics(intrinsic, original_size, self.image_size))

            sensor_to_ego = transform_from_quaternion(
                calibrated_sensor["rotation"], calibrated_sensor["translation"]
            )
            extrinsics.append(np.linalg.inv(sensor_to_ego).astype(np.float32))

        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_to_ego = transform_from_quaternion(lidar_cs["rotation"], lidar_cs["translation"])
        lidar_points = LidarPointCloud.from_file(
            str(self.dataroot / lidar_sd["filename"])
        ).points.T.astype(np.float32)
        lidar_xyz1 = np.concatenate(
            (lidar_points[:, :3], np.ones((lidar_points.shape[0], 1), dtype=np.float32)),
            axis=-1,
        )
        lidar_points[:, :3] = (lidar_xyz1 @ lidar_to_ego.T)[:, :3]

        ego_pose_record = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ego_to_global = transform_from_quaternion(
            ego_pose_record["rotation"], ego_pose_record["translation"]
        )
        global_to_ego = np.linalg.inv(ego_to_global)
        ego_yaw = yaw_from_quaternion(ego_pose_record["rotation"])

        boxes_3d: list[list[float]] = []
        labels: list[int] = []
        for ann_token in sample["anns"]:
            annotation = self.nusc.get("sample_annotation", ann_token)
            detection_name = category_to_detection_name(annotation["category_name"])
            if detection_name is None:
                continue
            center_global = np.asarray(annotation["translation"], dtype=np.float32)
            center_ego = (np.append(center_global, 1.0) @ global_to_ego.T)[:3]
            yaw_global = yaw_from_quaternion(annotation["rotation"])
            velocity_global = np.nan_to_num(
                self.nusc.box_velocity(ann_token)[:2], nan=0.0, posinf=0.0, neginf=0.0
            )
            velocity_ego = rotate_xy(velocity_global, -ego_yaw)
            size = np.asarray(annotation["size"], dtype=np.float32)
            boxes_3d.append(
                [
                    float(center_ego[0]),
                    float(center_ego[1]),
                    float(center_ego[2]),
                    float(size[0]),
                    float(size[1]),
                    float(size[2]),
                    wrap_angle(yaw_global - ego_yaw),
                    float(velocity_ego[0]),
                    float(velocity_ego[1]),
                ]
            )
            labels.append(NUSCENES_DETECTION_NAME_TO_INDEX[detection_name])

        od_targets = None
        if boxes_3d:
            od_targets = ObjectTargets(
                boxes_3d=torch.tensor(boxes_3d, dtype=torch.float32).unsqueeze(0),
                labels=torch.tensor(labels, dtype=torch.long).unsqueeze(0),
            )

        scene = SceneBatch(
            images=torch.stack(image_tensors, dim=0).unsqueeze(0),
            lidar_points=torch.from_numpy(lidar_points[:, :4]).unsqueeze(0),
            lidar_mask=torch.ones(1, lidar_points.shape[0], dtype=torch.bool),
            intrinsics=torch.from_numpy(np.stack(intrinsics, axis=0)).unsqueeze(0),
            extrinsics=torch.from_numpy(np.stack(extrinsics, axis=0)).unsqueeze(0),
            ego_pose=torch.from_numpy(ego_to_global).unsqueeze(0),
            time_delta_s=torch.zeros(1, dtype=torch.float32),
            od_targets=od_targets,
        )
        scene.validate()
        return SceneExample(
            scene=scene,
            metadata={
                "sample_token": sample_token,
                "ego_to_global": ego_to_global,
            },
        )


class OpenLaneDataset(Dataset[SceneExample]):
    """OpenLane 3D lane dataset in the canonical TSQBEV batch contract."""

    def __init__(
        self,
        dataroot: str | Path,
        split: str = "training",
        subset: str = "lane3d_300",
        image_size: tuple[int, int] = (256, 704),
        lane_points: int = 20,
    ) -> None:
        super().__init__()
        self.dataroot = Path(dataroot)
        self.split = split
        self.subset = subset
        self.image_size = image_size
        self.lane_points = lane_points
        lane_root = self.dataroot / subset / split
        self.annotation_paths = sorted(lane_root.rglob("*.json"))
        if not self.annotation_paths:
            raise FileNotFoundError(f"no OpenLane annotations found under {lane_root}")

    def __len__(self) -> int:
        return len(self.annotation_paths)

    def __getitem__(self, index: int) -> SceneExample:
        annotation_path = self.annotation_paths[index]
        annotation = json.loads(annotation_path.read_text())
        file_path = str(annotation["file_path"])

        image_path = self.dataroot / "images" / file_path
        image_tensor, original_size = _image_to_tensor(image_path, self.image_size)

        original_intrinsic = np.asarray(annotation["intrinsic"], dtype=np.float32)
        scaled_intrinsic = _scale_intrinsics(original_intrinsic, original_size, self.image_size)
        original_extrinsic = np.asarray(annotation["extrinsic"], dtype=np.float32)
        camera_to_ground = _openlane_camera_to_ground_extrinsic(original_extrinsic)
        ground_to_camera = np.linalg.inv(camera_to_ground).astype(np.float32)

        polylines: list[np.ndarray] = []
        for lane_line in annotation.get("lane_lines", []):
            xyz = np.asarray(lane_line["xyz"], dtype=np.float32).T
            lane_ground = _openlane_lane_points_to_ground(original_extrinsic, xyz)
            polylines.append(_resample_polyline_in_y(lane_ground, self.lane_points))

        lane_targets = None
        if polylines:
            lane_targets = LaneTargets(
                polylines=torch.from_numpy(np.stack(polylines, axis=0)).unsqueeze(0),
                valid_mask=torch.ones(1, len(polylines), dtype=torch.bool),
            )

        empty_lidar = torch.zeros(1, 1, 4, dtype=torch.float32)
        empty_lidar_mask = torch.zeros(1, 1, dtype=torch.bool)

        scene = SceneBatch(
            images=image_tensor.unsqueeze(0).unsqueeze(0),
            lidar_points=empty_lidar,
            lidar_mask=empty_lidar_mask,
            intrinsics=torch.from_numpy(scaled_intrinsic).unsqueeze(0).unsqueeze(0),
            extrinsics=torch.from_numpy(ground_to_camera).unsqueeze(0).unsqueeze(0),
            ego_pose=torch.eye(4, dtype=torch.float32).unsqueeze(0),
            time_delta_s=torch.zeros(1, dtype=torch.float32),
            lane_targets=lane_targets,
        )
        scene.validate()
        return SceneExample(
            scene=scene,
            metadata={
                "file_path": file_path,
                "intrinsic": original_intrinsic.tolist(),
                "extrinsic": original_extrinsic.tolist(),
            },
        )
