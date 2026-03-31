"""Compatibility shim for archived upstream stacks that only need light numba APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class _ErrorsModule:
    class NumbaPerformanceWarning(Warning):
        """Placeholder warning type used by archived upstream imports."""


errors = _ErrorsModule()


def _identity_decorator(*dargs: Any, **dkwargs: Any):
    if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
        return dargs[0]

    def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return _wrap


jit = _identity_decorator
njit = _identity_decorator
prange = range
