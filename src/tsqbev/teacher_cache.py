"""Cache helpers for optional external teacher targets.

References:
- BEVDistill:
  https://arxiv.org/abs/2211.09386
- OpenPCDet model zoo:
  https://github.com/open-mmlab/OpenPCDet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tsqbev.contracts import TeacherTargets

Tensor = torch.Tensor


def _clone_optional_tensor(tensor: Tensor | None) -> Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().clone()


@dataclass(slots=True)
class TeacherCacheRecord:
    """One serialized teacher-target record keyed by sample token or file path."""

    key: str
    backend: str
    targets: TeacherTargets
    metadata: dict[str, Any] | None = None


def teacher_targets_to_payload(targets: TeacherTargets) -> dict[str, Tensor | None]:
    """Convert teacher targets into a torch-save friendly payload."""

    return {
        "object_features": _clone_optional_tensor(targets.object_features),
        "object_boxes": _clone_optional_tensor(targets.object_boxes),
        "object_labels": _clone_optional_tensor(targets.object_labels),
        "object_scores": _clone_optional_tensor(targets.object_scores),
        "lane_features": _clone_optional_tensor(targets.lane_features),
        "router_logits": _clone_optional_tensor(targets.router_logits),
        "valid_mask": _clone_optional_tensor(targets.valid_mask),
    }


def teacher_targets_from_payload(payload: dict[str, Tensor | None]) -> TeacherTargets:
    """Reconstruct teacher targets from a serialized payload."""

    return TeacherTargets(
        object_features=payload.get("object_features"),
        object_boxes=payload.get("object_boxes"),
        object_labels=payload.get("object_labels"),
        object_scores=payload.get("object_scores"),
        lane_features=payload.get("lane_features"),
        router_logits=payload.get("router_logits"),
        valid_mask=payload.get("valid_mask"),
    )


class TeacherCacheStore:
    """Simple `.pt` record store for cached external teacher outputs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record_path(self, key: str) -> Path:
        return self.root / f"{key}.pt"

    def save(
        self,
        key: str,
        *,
        backend: str,
        targets: TeacherTargets,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        record = {
            "key": key,
            "backend": backend,
            "targets": teacher_targets_to_payload(targets),
            "metadata": metadata or {},
        }
        path = self.record_path(key)
        torch.save(record, path)
        return path

    def load(self, key: str) -> TeacherCacheRecord | None:
        path = self.record_path(key)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return TeacherCacheRecord(
            key=str(payload["key"]),
            backend=str(payload["backend"]),
            targets=teacher_targets_from_payload(payload["targets"]),
            metadata=dict(payload.get("metadata", {})),
        )
