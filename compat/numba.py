"""Compatibility shim for archived upstream stacks that only need JIT decorators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _identity_decorator(*dargs: Any, **dkwargs: Any):
    if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
        return dargs[0]

    def _wrap(func: F) -> F:
        return func

    return _wrap


jit = _identity_decorator
njit = _identity_decorator
