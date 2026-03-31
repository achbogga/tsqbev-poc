from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_sitecustomize_module():
    module_path = Path(__file__).resolve().parents[1] / "compat" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("tsqbev_compat_sitecustomize", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sitecustomize_restores_archived_numpy_aliases() -> None:
    _load_sitecustomize_module()

    assert np.long is np.int_
    assert np.bool is np.bool_
