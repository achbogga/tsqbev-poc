"""Optional external teacher backends for LiDAR-strong supervision.

References:
- OpenPCDet model zoo:
  https://github.com/open-mmlab/OpenPCDet
- CenterPoint:
  https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf
- BEVDistill:
  https://arxiv.org/abs/2211.09386
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from tsqbev.contracts import TeacherTargets
from tsqbev.teacher_cache import TeacherCacheStore

TeacherBackendKind = Literal[
    "cache",
    "openpcdet-centerpoint-pointpillar",
    "openpcdet-centerpoint-voxel",
]


def teacher_key_from_metadata(metadata: dict[str, Any]) -> str:
    """Resolve a stable cache key from canonical dataset metadata."""

    if "sample_token" in metadata:
        return str(metadata["sample_token"])
    if "file_path" in metadata:
        return str(metadata["file_path"]).replace("/", "__")
    raise KeyError("teacher metadata must include `sample_token` or `file_path`")


@dataclass(slots=True)
class TeacherProviderConfig:
    """Configuration for an optional external teacher source."""

    kind: TeacherBackendKind
    cache_dir: str | None = None
    checkpoint_path: str | None = None


class TeacherProvider(Protocol):
    """Contract for any external teacher source or teacher-cache reader."""

    kind: str

    def load_targets(self, metadata: dict[str, Any]) -> TeacherTargets | None:
        """Return cached or live teacher targets for one example."""


class CachedTeacherProvider:
    """Load teacher targets from a repo-local cache directory."""

    kind = "cache"

    def __init__(self, cache_dir: str) -> None:
        self.store = TeacherCacheStore(cache_dir)

    def load_targets(self, metadata: dict[str, Any]) -> TeacherTargets | None:
        record = self.store.load(teacher_key_from_metadata(metadata))
        if record is None:
            return None
        return record.targets


class OpenPCDetCenterPointProvider:
    """Optional OpenPCDet-backed CenterPoint teacher provider.

    This adapter is intentionally conservative in the public repo. The core runtime
    remains dependency-light, so live OpenPCDet execution is not wired into the
    default environment. The expected path is:

    1. run the external teacher in a separate environment,
    2. cache teacher targets with `TeacherCacheStore`,
    3. train tsqbev-poc from the cache.
    """

    def __init__(self, config: TeacherProviderConfig) -> None:
        self.kind: str = str(config.kind)
        self.checkpoint_path = config.checkpoint_path
        if self.checkpoint_path is None:
            raise ValueError("OpenPCDet teacher providers require `checkpoint_path`")

    def load_targets(self, metadata: dict[str, Any]) -> TeacherTargets | None:
        del metadata
        raise RuntimeError(
            "Live OpenPCDet teacher execution is intentionally not embedded into the "
            "core repo runtime. Run the external CenterPoint model in its own "
            "environment and cache teacher targets via TeacherCacheStore."
        )


def build_teacher_provider(config: TeacherProviderConfig) -> TeacherProvider:
    """Construct an optional teacher provider or cache reader."""

    if config.kind == "cache":
        if config.cache_dir is None:
            raise ValueError("cache teacher provider requires `cache_dir`")
        return CachedTeacherProvider(config.cache_dir)
    if config.kind in {
        "openpcdet-centerpoint-pointpillar",
        "openpcdet-centerpoint-voxel",
    }:
        return OpenPCDetCenterPointProvider(config)
    raise ValueError(f"unsupported teacher provider kind: {config.kind}")
