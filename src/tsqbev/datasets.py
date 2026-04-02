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
from typing import Any, Literal, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from tsqbev.contracts import CameraProposals, LaneTargets, ObjectTargets, SceneBatch, TeacherTargets
from tsqbev.labels import (
    NUSCENES_CAMERA_NAMES,
    NUSCENES_DETECTION_NAME_TO_INDEX,
)
from tsqbev.quaternion import rotate_xy, transform_from_quaternion, wrap_angle, yaw_from_quaternion

Tensor = torch.Tensor
AugmentationMode = Literal["off", "moderate", "strong"]

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


def _is_training_split(split: str) -> bool:
    return split in {"train", "mini_train", "training"}


def _augmentation_strength(mode: AugmentationMode) -> dict[str, float]:
    if mode == "strong":
        return {
            "brightness": 0.20,
            "contrast": 0.20,
            "saturation": 0.20,
            "hue": 0.05,
            "blur_prob": 0.20,
            "noise_std": 0.03,
            "erase_prob": 0.15,
            "erase_area_min": 0.02,
            "erase_area_max": 0.12,
            "lidar_dropout": 0.10,
            "lidar_xyz_jitter": 0.08,
            "lidar_intensity_jitter": 0.04,
        }
    return {
        "brightness": 0.12,
        "contrast": 0.12,
        "saturation": 0.12,
        "hue": 0.03,
        "blur_prob": 0.10,
        "noise_std": 0.015,
        "erase_prob": 0.05,
        "erase_area_min": 0.02,
        "erase_area_max": 0.08,
        "lidar_dropout": 0.05,
        "lidar_xyz_jitter": 0.04,
        "lidar_intensity_jitter": 0.02,
    }


def _apply_random_erasing(
    image: Tensor,
    *,
    area_min: float,
    area_max: float,
) -> Tensor:
    channels, height, width = image.shape
    del channels
    image_area = float(height * width)
    erase_fraction = float(torch.empty(1).uniform_(area_min, area_max).item())
    erase_area = max(1, int(image_area * erase_fraction))
    aspect_ratio = float(torch.empty(1).uniform_(0.5, 2.0).item())
    erase_h = min(height, max(1, int(round((erase_area / aspect_ratio) ** 0.5))))
    erase_w = min(width, max(1, int(round((erase_area * aspect_ratio) ** 0.5))))
    if erase_h >= height:
        top = 0
    else:
        top = int(torch.randint(0, height - erase_h + 1, (1,)).item())
    if erase_w >= width:
        left = 0
    else:
        left = int(torch.randint(0, width - erase_w + 1, (1,)).item())
    image = image.clone()
    image[:, top : top + erase_h, left : left + erase_w] = image.mean(dim=(-1, -2), keepdim=True)
    return image


def _apply_image_augmentations(image: Tensor, mode: AugmentationMode) -> Tensor:
    if mode == "off":
        return image
    strength = _augmentation_strength(mode)
    augmented = image.clone()

    brightness = 1.0 + float(
        torch.empty(1).uniform_(-strength["brightness"], strength["brightness"]).item()
    )
    contrast = 1.0 + float(
        torch.empty(1).uniform_(-strength["contrast"], strength["contrast"]).item()
    )
    saturation = 1.0 + float(
        torch.empty(1).uniform_(-strength["saturation"], strength["saturation"]).item()
    )
    hue = float(torch.empty(1).uniform_(-strength["hue"], strength["hue"]).item())
    augmented = TF.adjust_brightness(augmented, brightness)
    augmented = TF.adjust_contrast(augmented, contrast)
    augmented = TF.adjust_saturation(augmented, saturation)
    augmented = TF.adjust_hue(augmented, hue)

    if torch.rand(()) < strength["blur_prob"]:
        sigma = float(torch.empty(1).uniform_(0.1, 1.1).item())
        augmented = TF.gaussian_blur(augmented, kernel_size=[5, 5], sigma=[sigma, sigma])

    if strength["noise_std"] > 0.0:
        augmented = augmented + torch.randn_like(augmented) * strength["noise_std"]

    if torch.rand(()) < strength["erase_prob"]:
        augmented = _apply_random_erasing(
            augmented,
            area_min=strength["erase_area_min"],
            area_max=strength["erase_area_max"],
        )
    return augmented.clamp(0.0, 1.0)


def _apply_lidar_augmentations(
    points: Tensor,
    mask: Tensor,
    mode: AugmentationMode,
) -> tuple[Tensor, Tensor]:
    if mode == "off" or not bool(mask.any()):
        return points, mask
    strength = _augmentation_strength(mode)
    augmented_points = points.clone()
    augmented_mask = mask.clone()
    keep_mask = torch.rand_like(augmented_mask.float()) > strength["lidar_dropout"]
    augmented_mask = augmented_mask & keep_mask
    if bool(augmented_mask.any()):
        augmented_points[augmented_mask, :3] += (
            torch.randn_like(augmented_points[augmented_mask, :3]) * strength["lidar_xyz_jitter"]
        )
        augmented_points[augmented_mask, 3] += (
            torch.randn_like(augmented_points[augmented_mask, 3])
            * strength["lidar_intensity_jitter"]
        )
    augmented_points[~augmented_mask] = 0.0
    return augmented_points, augmented_mask


def _pad_rows(tensor: Tensor, target_rows: int) -> Tensor:
    rows = int(tensor.shape[0])
    if rows == target_rows:
        return tensor
    padded_shape = (target_rows, *tensor.shape[1:])
    padded = tensor.new_zeros(padded_shape)
    padded[:rows] = tensor
    return padded


def _collate_object_targets(examples: list[SceneExample]) -> ObjectTargets | None:
    max_objects = max(
        (
            int(example.scene.od_targets.boxes_3d.shape[1])
            if example.scene.od_targets is not None
            else 0
        )
        for example in examples
    )
    if max_objects == 0:
        return None

    boxes: list[Tensor] = []
    labels: list[Tensor] = []
    valid_masks: list[Tensor] = []
    for example in examples:
        targets = example.scene.od_targets
        if targets is None:
            boxes.append(example.scene.images.new_zeros(max_objects, 9))
            labels.append(torch.zeros(max_objects, dtype=torch.long))
            valid_masks.append(torch.zeros(max_objects, dtype=torch.bool))
            continue
        boxes_3d = targets.boxes_3d.squeeze(0)
        label_ids = targets.labels.squeeze(0)
        valid_mask = targets.valid_mask.squeeze(0)
        boxes.append(_pad_rows(boxes_3d, max_objects))
        labels.append(_pad_rows(label_ids, max_objects))
        valid_masks.append(_pad_rows(valid_mask, max_objects))

    return ObjectTargets(
        boxes_3d=torch.stack(boxes, dim=0),
        labels=torch.stack(labels, dim=0),
        valid_mask=torch.stack(valid_masks, dim=0),
    )


def _collate_lane_targets(examples: list[SceneExample]) -> LaneTargets | None:
    max_lanes = max(
        (
            int(example.scene.lane_targets.polylines.shape[1])
            if example.scene.lane_targets is not None
            else 0
        )
        for example in examples
    )
    if max_lanes == 0:
        return None

    lane_points = max(
        (
            int(example.scene.lane_targets.polylines.shape[2])
            if example.scene.lane_targets is not None
            else 0
        )
        for example in examples
    )
    polylines: list[Tensor] = []
    valid_masks: list[Tensor] = []
    for example in examples:
        targets = example.scene.lane_targets
        if targets is None:
            polylines.append(example.scene.images.new_zeros(max_lanes, lane_points, 3))
            valid_masks.append(torch.zeros(max_lanes, dtype=torch.bool))
            continue
        lane_polylines = targets.polylines.squeeze(0)
        valid_mask = targets.valid_mask.squeeze(0)
        padded_polylines = lane_polylines.new_zeros(max_lanes, lane_points, 3)
        padded_polylines[: lane_polylines.shape[0], : lane_polylines.shape[1]] = lane_polylines
        polylines.append(padded_polylines)
        valid_masks.append(_pad_rows(valid_mask, max_lanes))

    return LaneTargets(
        polylines=torch.stack(polylines, dim=0),
        valid_mask=torch.stack(valid_masks, dim=0),
    )


def _collate_camera_proposals(examples: list[SceneExample]) -> CameraProposals | None:
    proposals = [example.scene.camera_proposals for example in examples]
    if all(proposal is None for proposal in proposals):
        return None
    if any(proposal is None for proposal in proposals):
        raise ValueError("camera proposals must be present for every example or none")
    concrete_proposals = [cast(CameraProposals, proposal) for proposal in proposals]
    return CameraProposals(
        boxes_xyxy=torch.cat([proposal.boxes_xyxy for proposal in concrete_proposals], dim=0),
        scores=torch.cat([proposal.scores for proposal in concrete_proposals], dim=0),
    )


def _max_teacher_rows(
    examples: list[SceneExample],
    *field_names: str,
) -> int:
    max_rows = 0
    for example in examples:
        targets = example.scene.teacher_targets
        if targets is None:
            continue
        for field_name in field_names:
            field = getattr(targets, field_name)
            if field is not None:
                max_rows = max(max_rows, int(field.shape[1]))
    return max_rows


def _teacher_field_present(examples: list[SceneExample], field_name: str) -> bool:
    for example in examples:
        targets = example.scene.teacher_targets
        if targets is not None and getattr(targets, field_name) is not None:
            return True
    return False


def _teacher_feature_dim(examples: list[SceneExample], field_name: str) -> int:
    for example in examples:
        targets = example.scene.teacher_targets
        if targets is None:
            continue
        field = getattr(targets, field_name)
        if field is not None:
            return int(field.shape[-1])
    raise ValueError(f"teacher field {field_name} is not present in any example")


def _collate_teacher_targets(examples: list[SceneExample]) -> TeacherTargets | None:
    if all(example.scene.teacher_targets is None for example in examples):
        return None

    max_object_rows = _max_teacher_rows(
        examples,
        "object_features",
        "object_boxes",
        "object_labels",
        "object_scores",
        "router_logits",
        "valid_mask",
    )
    max_lane_rows = _max_teacher_rows(examples, "lane_features")
    if max_object_rows == 0 and max_lane_rows == 0:
        return None

    object_features: list[Tensor] = []
    object_boxes: list[Tensor] = []
    object_labels: list[Tensor] = []
    object_scores: list[Tensor] = []
    lane_features: list[Tensor] = []
    router_logits: list[Tensor] = []
    valid_masks: list[Tensor] = []

    has_object_features = _teacher_field_present(examples, "object_features")
    has_object_boxes = _teacher_field_present(examples, "object_boxes")
    has_object_labels = _teacher_field_present(examples, "object_labels")
    has_object_scores = _teacher_field_present(examples, "object_scores")
    has_lane_features = _teacher_field_present(examples, "lane_features")
    has_router_logits = _teacher_field_present(examples, "router_logits")
    has_valid_mask = _teacher_field_present(examples, "valid_mask")

    object_feature_dim = (
        _teacher_feature_dim(examples, "object_features") if has_object_features else 0
    )
    lane_feature_dim = _teacher_feature_dim(examples, "lane_features") if has_lane_features else 0

    for example in examples:
        scene = example.scene
        targets = scene.teacher_targets
        image_tensor = scene.images

        if has_object_features:
            if targets is None or targets.object_features is None:
                object_features.append(
                    image_tensor.new_zeros(max_object_rows, object_feature_dim)
                )
            else:
                object_features.append(
                    _pad_rows(targets.object_features.squeeze(0), max_object_rows)
                )
        if has_object_boxes:
            if targets is None or targets.object_boxes is None:
                object_boxes.append(image_tensor.new_zeros(max_object_rows, 9))
            else:
                object_boxes.append(_pad_rows(targets.object_boxes.squeeze(0), max_object_rows))
        if has_object_labels:
            if targets is None or targets.object_labels is None:
                object_labels.append(torch.zeros(max_object_rows, dtype=torch.long))
            else:
                object_labels.append(_pad_rows(targets.object_labels.squeeze(0), max_object_rows))
        if has_object_scores:
            if targets is None or targets.object_scores is None:
                object_scores.append(image_tensor.new_zeros(max_object_rows))
            else:
                object_scores.append(_pad_rows(targets.object_scores.squeeze(0), max_object_rows))
        if has_lane_features:
            if targets is None or targets.lane_features is None:
                lane_features.append(image_tensor.new_zeros(max_lane_rows, lane_feature_dim))
            else:
                lane_features.append(_pad_rows(targets.lane_features.squeeze(0), max_lane_rows))
        if has_router_logits:
            if targets is None or targets.router_logits is None:
                router_logits.append(image_tensor.new_zeros(max_object_rows))
            else:
                router_logits.append(_pad_rows(targets.router_logits.squeeze(0), max_object_rows))
        if has_valid_mask or max_object_rows > 0:
            if targets is None or targets.valid_mask is None:
                valid_masks.append(torch.zeros(max_object_rows, dtype=torch.bool))
            else:
                valid_masks.append(_pad_rows(targets.valid_mask.squeeze(0), max_object_rows))

    collated = TeacherTargets(
        object_features=torch.stack(object_features, dim=0) if has_object_features else None,
        object_boxes=torch.stack(object_boxes, dim=0) if has_object_boxes else None,
        object_labels=torch.stack(object_labels, dim=0) if has_object_labels else None,
        object_scores=torch.stack(object_scores, dim=0) if has_object_scores else None,
        lane_features=torch.stack(lane_features, dim=0) if has_lane_features else None,
        router_logits=torch.stack(router_logits, dim=0) if has_router_logits else None,
        valid_mask=torch.stack(valid_masks, dim=0) if valid_masks else None,
    )
    collated.validate(len(examples))
    return collated


def collate_scene_examples(examples: list[SceneExample]) -> tuple[SceneBatch, list[dict[str, Any]]]:
    """Collate real dataset examples into a padded batch."""

    if not examples:
        raise ValueError("cannot collate an empty example list")

    max_points = max(int(example.scene.lidar_points.shape[1]) for example in examples)
    lidar_points: list[Tensor] = []
    lidar_masks: list[Tensor] = []
    for example in examples:
        scene = example.scene
        points = scene.lidar_points.squeeze(0)
        mask = scene.lidar_mask.squeeze(0)
        lidar_points.append(_pad_rows(points, max_points))
        lidar_masks.append(_pad_rows(mask, max_points))

    batch = SceneBatch(
        images=torch.cat([example.scene.images for example in examples], dim=0),
        lidar_points=torch.stack(lidar_points, dim=0),
        lidar_mask=torch.stack(lidar_masks, dim=0),
        intrinsics=torch.cat([example.scene.intrinsics for example in examples], dim=0),
        extrinsics=torch.cat([example.scene.extrinsics for example in examples], dim=0),
        ego_pose=torch.cat([example.scene.ego_pose for example in examples], dim=0),
        time_delta_s=torch.cat([example.scene.time_delta_s for example in examples], dim=0),
        camera_proposals=_collate_camera_proposals(examples),
        od_targets=_collate_object_targets(examples),
        lane_targets=_collate_lane_targets(examples),
        teacher_targets=_collate_teacher_targets(examples),
    )
    batch.validate()
    return batch, [example.metadata for example in examples]


def collate_single_scene_example(
    examples: list[SceneExample],
) -> tuple[SceneBatch, list[dict[str, Any]]]:
    """Backward-compatible alias for the real padded batch collator."""

    return collate_scene_examples(examples)


def _image_to_tensor(path: Path, image_size: tuple[int, int]) -> tuple[Tensor, tuple[int, int]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        original_size = rgb.height, rgb.width
        resized = rgb.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
        tensor = TF.pil_to_tensor(resized).float() / 255.0
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
        sample_tokens: list[str] | None = None,
        augmentation_mode: AugmentationMode = "off",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        NuScenes, _, _, create_splits_scenes = _load_nuscenes_devkit()
        self.dataroot = Path(dataroot)
        self.version = version
        self.split = split
        self.image_size = image_size
        self.augmentation_mode = augmentation_mode if _is_training_split(split) else "off"
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=verbose)
        if sample_tokens is not None:
            self.sample_tokens = list(sample_tokens)
        else:
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
            image_tensor = _apply_image_augmentations(image_tensor, self.augmentation_mode)
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
        lidar_tensor = torch.from_numpy(lidar_points[:, :4])
        lidar_mask = torch.ones(lidar_points.shape[0], dtype=torch.bool)
        lidar_tensor, lidar_mask = _apply_lidar_augmentations(
            lidar_tensor,
            lidar_mask,
            self.augmentation_mode,
        )

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
                valid_mask=torch.ones(1, len(boxes_3d), dtype=torch.bool),
            )

        scene = SceneBatch(
            images=torch.stack(image_tensors, dim=0).unsqueeze(0),
            lidar_points=lidar_tensor.unsqueeze(0),
            lidar_mask=lidar_mask.unsqueeze(0),
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
        augmentation_mode: AugmentationMode = "off",
    ) -> None:
        super().__init__()
        self.dataroot = Path(dataroot)
        self.split = split
        self.subset = subset
        self.image_size = image_size
        self.lane_points = lane_points
        self.augmentation_mode = augmentation_mode if _is_training_split(split) else "off"
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
        image_tensor = _apply_image_augmentations(image_tensor, self.augmentation_mode)

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
