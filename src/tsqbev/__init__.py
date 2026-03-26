"""tsqbev-poc package.

References:
- DETR3D paper: https://proceedings.mlr.press/v164/wang22b/wang22b.pdf
- PETR official repo: https://github.com/megvii-research/PETR
- Sparse4D official repo: https://github.com/HorizonRobotics/Sparse4D
- HotBEV paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf
"""

from tsqbev.config import ModelConfig
from tsqbev.model import TSQBEVCore, TSQBEVModel

__all__ = ["ModelConfig", "TSQBEVCore", "TSQBEVModel"]
