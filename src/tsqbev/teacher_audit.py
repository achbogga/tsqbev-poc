"""Audit coverage of cached external teacher targets against nuScenes splits.

References:
- nuScenes devkit and split structure:
  https://github.com/nutonomy/nuscenes-devkit
- OpenPCDet nuScenes result export:
  https://github.com/open-mmlab/OpenPCDet
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tsqbev.datasets import NuScenesDataset
from tsqbev.teacher_cache import TeacherCacheStore


def audit_nuscenes_teacher_cache(
    dataroot: str | Path,
    version: str,
    split: str,
    cache_dir: str | Path,
    *,
    max_samples: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Measure teacher-cache coverage for one nuScenes split."""

    dataset = NuScenesDataset(dataroot=dataroot, version=version, split=split)
    sample_tokens = list(dataset.sample_tokens)
    if max_samples is not None:
        sample_tokens = sample_tokens[:max_samples]

    store = TeacherCacheStore(cache_dir)
    present_tokens: list[str] = []
    missing_tokens: list[str] = []
    for sample_token in sample_tokens:
        if store.record_path(sample_token).exists():
            present_tokens.append(sample_token)
        else:
            missing_tokens.append(sample_token)

    total = len(sample_tokens)
    present = len(present_tokens)
    coverage = float(present / total) if total else 0.0
    summary = {
        "status": "completed",
        "version": version,
        "split": split,
        "cache_dir": str(cache_dir),
        "total_samples": total,
        "present_records": present,
        "missing_records": len(missing_tokens),
        "coverage": coverage,
        "present_sample_tokens_head": present_tokens[:16],
        "missing_sample_tokens_head": missing_tokens[:32],
    }

    if output_dir is not None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
        (output_root / "missing_sample_tokens.json").write_text(
            json.dumps(missing_tokens, indent=2)
        )

    return summary
