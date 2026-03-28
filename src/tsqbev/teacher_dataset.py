"""Dataset wrappers for attaching cached external teacher targets.

References:
- BEVDistill:
  https://arxiv.org/abs/2211.09386
- OpenPCDet:
  https://github.com/open-mmlab/OpenPCDet
"""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import replace
from typing import Protocol, cast

from torch.utils.data import Dataset

from tsqbev.contracts import TeacherTargets
from tsqbev.datasets import SceneExample


class _TeacherProviderLike(Protocol):
    def load_targets(self, metadata: dict[str, object]) -> TeacherTargets | None:
        """Return teacher targets for one example."""


def attach_teacher_targets(
    example: SceneExample,
    teacher_targets: TeacherTargets | None,
) -> SceneExample:
    """Return a new scene example with teacher targets attached."""

    if teacher_targets is None:
        return example
    scene = replace(example.scene, teacher_targets=teacher_targets)
    scene.validate()
    return SceneExample(scene=scene, metadata=example.metadata)


class TeacherAugmentedDataset(Dataset[SceneExample]):
    """Wrap a base dataset and inject teacher targets from a provider."""

    def __init__(
        self,
        base_dataset: Dataset[SceneExample],
        provider: _TeacherProviderLike,
    ) -> None:
        self.base_dataset = base_dataset
        self.provider = provider

    def __len__(self) -> int:
        return len(cast(Sized, self.base_dataset))

    def __getitem__(self, index: int) -> SceneExample:
        example = self.base_dataset[index]
        teacher_targets = self.provider.load_targets(example.metadata)
        return attach_teacher_targets(example, teacher_targets)
