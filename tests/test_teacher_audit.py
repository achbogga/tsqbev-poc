from __future__ import annotations

import json

import torch

from tsqbev.contracts import TeacherTargets
from tsqbev.teacher_audit import audit_nuscenes_teacher_cache
from tsqbev.teacher_cache import TeacherCacheStore


class _FakeNuScenesDataset:
    def __init__(self, dataroot: str, version: str, split: str) -> None:
        del dataroot, version, split
        self.sample_tokens = ["a", "b", "c", "d"]


def test_audit_nuscenes_teacher_cache_reports_coverage(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "tsqbev.teacher_audit.NuScenesDataset",
        _FakeNuScenesDataset,
    )
    store = TeacherCacheStore(tmp_path / "cache")
    empty = TeacherTargets(
        object_boxes=torch.zeros(1, 0, 9),
        object_labels=torch.zeros(1, 0, dtype=torch.long),
        object_scores=torch.zeros(1, 0),
        valid_mask=torch.zeros(1, 0, dtype=torch.bool),
    )
    store.save("a", backend="test", targets=empty, metadata={"sample_token": "a"})
    store.save("c", backend="test", targets=empty, metadata={"sample_token": "c"})

    summary = audit_nuscenes_teacher_cache(
        dataroot="/unused",
        version="v1.0-mini",
        split="mini_train",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "audit",
    )

    assert summary["total_samples"] == 4
    assert summary["present_records"] == 2
    assert summary["missing_records"] == 2
    assert summary["coverage"] == 0.5
    assert summary["missing_sample_tokens_head"] == ["b", "d"]
    written = json.loads((tmp_path / "audit" / "summary.json").read_text())
    assert written["coverage"] == 0.5
