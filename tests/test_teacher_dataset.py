from __future__ import annotations

import torch

from tsqbev.contracts import TeacherTargets
from tsqbev.datasets import SceneExample
from tsqbev.synthetic import make_synthetic_batch
from tsqbev.teacher_dataset import TeacherAugmentedDataset


class _StaticTeacherProvider:
    def __init__(self, targets: TeacherTargets) -> None:
        self.targets = targets

    def load_targets(self, metadata: dict[str, object]) -> TeacherTargets | None:
        assert metadata["sample_token"] == "sample-1"
        return self.targets


def test_teacher_augmented_dataset_attaches_targets(small_config) -> None:
    scene = make_synthetic_batch(small_config, batch_size=1, with_teacher=False)
    example = SceneExample(scene=scene, metadata={"sample_token": "sample-1"})
    targets = TeacherTargets(
        object_boxes=torch.randn(1, 2, 9),
        object_labels=torch.tensor([[0, 1]], dtype=torch.long),
        object_scores=torch.tensor([[0.9, 0.5]], dtype=torch.float32),
        valid_mask=torch.tensor([[True, True]]),
    )

    dataset = TeacherAugmentedDataset([example], _StaticTeacherProvider(targets))
    augmented = dataset[0]

    assert augmented.scene.teacher_targets is not None
    assert torch.allclose(augmented.scene.teacher_targets.object_boxes, targets.object_boxes)
