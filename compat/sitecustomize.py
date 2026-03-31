"""Process-wide compatibility shims for archived upstream stacks."""

from __future__ import annotations

import numpy as np

if "long" not in np.__dict__:
    np.long = np.int_

if "bool" not in np.__dict__:
    np.bool = np.bool_
