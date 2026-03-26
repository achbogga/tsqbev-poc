from __future__ import annotations

import pytest

from tsqbev.research_guard import ensure_research_loop_disabled


def test_research_loop_is_disabled() -> None:
    with pytest.raises(RuntimeError):
        ensure_research_loop_disabled()
