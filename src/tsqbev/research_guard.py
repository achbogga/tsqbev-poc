"""Guard rails for the bounded local research loop.

References:
- Karpathy autoresearch workflow template:
  https://github.com/karpathy/autoresearch
"""

from __future__ import annotations

from pathlib import Path

PROGRAM_PATH = Path(__file__).resolve().parents[2] / "program.md"


def _read_status_line(program_path: str | Path = PROGRAM_PATH) -> str:
    path = Path(program_path)
    if not path.exists():
        raise RuntimeError(f"program file not found: {path}")
    for line in path.read_text().splitlines():
        if line.startswith("Status:"):
            return line.split(":", maxsplit=1)[1].strip().rstrip(".")
    raise RuntimeError(f"program file does not define a Status line: {path}")


def ensure_research_loop_enabled(program_path: str | Path = PROGRAM_PATH) -> None:
    """Raise unless the program explicitly enables the bounded research loop."""

    status = _read_status_line(program_path)
    if status != "enabled":
        raise RuntimeError(
            "Autonomous research is not enabled in program.md. "
            "Set Status: enabled only when the bounded loop is intentionally authorized."
        )
