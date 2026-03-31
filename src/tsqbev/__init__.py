"""tsqbev-poc package.

References:
- BEVFusion repo: https://github.com/mit-han-lab/bevfusion
- OpenPCDet repo: https://github.com/open-mmlab/OpenPCDet
- BEVDet repo: https://github.com/HuangJunJie2017/BEVDet
- MapTR repo: https://github.com/hustvl/MapTR
"""

from tsqbev.bevfusion_env import (
    check_bevfusion_environment,
    render_bevfusion_runbook_markdown,
)
from tsqbev.config import ModelConfig
from tsqbev.gap_analysis import analyze_reset_gap
from tsqbev.model import TSQBEVCore, TSQBEVModel
from tsqbev.reset_stack import recommended_reset_plan

__all__ = [
    "ModelConfig",
    "TSQBEVCore",
    "TSQBEVModel",
    "analyze_reset_gap",
    "check_bevfusion_environment",
    "render_bevfusion_runbook_markdown",
    "recommended_reset_plan",
]
