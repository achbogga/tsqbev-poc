from __future__ import annotations

import pytest

from tsqbev.research_guard import ensure_research_loop_enabled


def test_research_loop_requires_enabled_status(tmp_path) -> None:
    program = tmp_path / "program.md"
    program.write_text("Status: disabled.\n")
    with pytest.raises(RuntimeError):
        ensure_research_loop_enabled(program)


def test_research_loop_accepts_enabled_status(tmp_path) -> None:
    program = tmp_path / "program.md"
    program.write_text("Status: enabled.\n")
    ensure_research_loop_enabled(program)
