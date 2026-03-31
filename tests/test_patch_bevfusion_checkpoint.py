from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_patch_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "scripts"
        / "patch_bevfusion_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("patch_bevfusion_checkpoint", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_state_dict_expands_scalar_depth_stem_with_zeroed_feature_channels() -> None:
    module = _load_patch_module()
    state_dict = {
        module.DEPTH_STEM_KEY: torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1),
    }

    changed = module._patch_state_dict(state_dict, target_depth_channels=6)

    assert changed is True
    patched = state_dict[module.DEPTH_STEM_KEY]
    assert tuple(patched.shape) == (8, 6, 1, 1)
    assert torch.equal(patched[:, :1], torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1))
    assert torch.count_nonzero(patched[:, 1:]) == 0


def test_patch_state_dict_noops_when_shape_already_matches() -> None:
    module = _load_patch_module()
    original = torch.ones((8, 6, 1, 1), dtype=torch.float32)
    state_dict = {module.DEPTH_STEM_KEY: original.clone()}

    changed = module._patch_state_dict(state_dict, target_depth_channels=6)

    assert changed is False
    assert torch.equal(state_dict[module.DEPTH_STEM_KEY], original)
