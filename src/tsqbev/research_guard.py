"""Guard rails for intentionally disabled research automation.

References:
- Karpathy autoresearch workflow template:
  https://github.com/karpathy/autoresearch
"""

from __future__ import annotations


def ensure_research_loop_disabled() -> None:
    """Raise an error because bootstrap has not authorized the loop yet."""

    raise RuntimeError(
        "Autonomous research is disabled. "
        "Make the repo functional first, then explicitly enable bounded research later."
    )
