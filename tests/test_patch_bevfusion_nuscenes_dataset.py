from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_patch_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "scripts"
        / "patch_bevfusion_nuscenes_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("patch_bevfusion_nuscenes_dataset", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_snippets_cover_expected_archived_contract() -> None:
    module = _load_patch_module()
    assert "radar=info.get('radars', None)" in module.OLD_DATA_BLOCK
    assert "if self._needs_radar_pipeline_input()" in module.NEW_DATA_BLOCK
    assert "LoadRadarPointsMultiSweeps" in module.HELPER_BLOCK
