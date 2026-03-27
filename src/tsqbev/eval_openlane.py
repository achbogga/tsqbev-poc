"""OpenLane prediction export and official evaluation wrapper.

References:
- OpenLane official 3D evaluation format:
  https://github.com/OpenDriveLab/OpenLane/blob/main/eval/LANE_evaluation/README.md
- OpenLane official 3D lane evaluator:
  https://github.com/OpenDriveLab/OpenLane/blob/main/eval/LANE_evaluation/lane3d/eval_3D_lane.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tsqbev.datasets import OpenLaneDataset, collate_single_scene_example
from tsqbev.labels import OPENLANE_DEFAULT_CATEGORY
from tsqbev.model import TSQBEVModel
from tsqbev.runtime import move_batch, resolve_device


@torch.no_grad()
def export_openlane_predictions(
    model: TSQBEVModel,
    dataroot: str | Path,
    output_dir: str | Path,
    split: str = "validation",
    subset: str = "lane3d_300",
    score_threshold: float = 0.5,
    max_lanes: int = 64,
    device: str | None = None,
) -> Path:
    """Write OpenLane prediction JSON files under the official directory layout."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    dataset = OpenLaneDataset(dataroot=dataroot, split=split, subset=subset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_single_scene_example,
        pin_memory=torch.cuda.is_available(),
    )

    model = model.to(resolved_device).eval()
    for batch, metadata_list in loader:
        metadata = metadata_list[0]
        batch = move_batch(batch, resolved_device)
        outputs = model(batch)
        lane_logits = outputs["lane_logits"]
        lane_polylines = outputs["lane_polylines"]
        assert isinstance(lane_logits, torch.Tensor)
        assert isinstance(lane_polylines, torch.Tensor)

        probabilities = lane_logits[0].sigmoid()
        order = torch.argsort(probabilities, descending=True)
        lane_lines: list[dict[str, object]] = []
        for lane_index in order[:max_lanes]:
            if float(probabilities[lane_index]) < score_threshold:
                continue
            polyline = lane_polylines[0, lane_index].detach().cpu().numpy().T.astype(float).tolist()
            lane_lines.append({"xyz": polyline, "category": OPENLANE_DEFAULT_CATEGORY})

        file_path = Path(str(metadata["file_path"]))
        result_path = output_dir / file_path.with_suffix(".json")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "intrinsic": metadata["intrinsic"],
            "extrinsic": metadata["extrinsic"],
            "file_path": str(metadata["file_path"]),
            "lane_lines": lane_lines,
        }
        result_path.write_text(json.dumps(payload))

    return output_dir


def evaluate_openlane_predictions(
    openlane_repo_root: str | Path,
    dataset_dir: str | Path,
    pred_dir: str | Path,
    test_list: str | Path,
) -> dict[str, float | str]:
    """Run the official OpenLane 3D lane evaluator from a checked-out public repo."""

    repo_root = Path(openlane_repo_root)
    evaluator = repo_root / "eval" / "LANE_evaluation" / "lane3d" / "eval_3D_lane.py"
    if not evaluator.exists():
        raise FileNotFoundError(f"OpenLane evaluator not found at {evaluator}")

    command = [
        "python",
        str(evaluator),
        f"--dataset_dir={dataset_dir}",
        f"--pred_dir={pred_dir}",
        f"--test_list={test_list}",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    metrics: dict[str, float | str] = {"stdout": result.stdout}
    pattern = re.compile(
        r"F-measure ([0-9.eE+-]+).*?Recall\s+([0-9.eE+-]+).*?Precision\s+([0-9.eE+-]+)"
        r".*?Category Accuracy\s+([0-9.eE+-]+).*?x error \(close\)\s+([0-9.eE+-]+)"
        r".*?x error \(far\)\s+([0-9.eE+-]+).*?z error \(close\)\s+([0-9.eE+-]+)"
        r".*?z error \(far\)\s+([0-9.eE+-]+)",
        re.S,
    )
    match = pattern.search(result.stdout)
    if match is not None:
        metrics.update(
            {
                "f_score": float(match.group(1)),
                "recall": float(match.group(2)),
                "precision": float(match.group(3)),
                "category_accuracy": float(match.group(4)),
                "x_error_close_m": float(match.group(5)),
                "x_error_far_m": float(match.group(6)),
                "z_error_close_m": float(match.group(7)),
                "z_error_far_m": float(match.group(8)),
            }
        )
    return metrics
