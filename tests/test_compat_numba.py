from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_numba_compat():
    module_path = Path(__file__).resolve().parents[1] / "compat" / "numba.py"
    spec = importlib.util.spec_from_file_location("tsqbev_compat_numba", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numba_compat_decorators_are_identity() -> None:
    numba = _load_numba_compat()

    @numba.jit(nopython=True)
    def add_one(x: int) -> int:
        return x + 1

    @numba.njit
    def add_two(x: int) -> int:
        return x + 2

    assert add_one(2) == 3
    assert add_two(2) == 4


def test_numba_compat_exposes_expected_errors_namespace() -> None:
    numba = _load_numba_compat()
    assert issubclass(numba.errors.NumbaPerformanceWarning, Warning)
    assert list(numba.prange(3)) == [0, 1, 2]
