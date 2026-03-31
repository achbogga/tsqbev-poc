"""Process-wide compatibility shims for archived upstream stacks."""

from __future__ import annotations

import numpy as np

if not hasattr(np, "long"):
    np.long = np.int_
