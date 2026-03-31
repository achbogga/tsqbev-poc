"""Gap analysis between the legacy sparse-query repo and the dense-BEV reset target.

References:
- BEVFusion unified BEV design:
  https://github.com/mit-han-lab/bevfusion
- OpenPCDet public model zoo:
  https://github.com/open-mmlab/OpenPCDet
- BEVDet temporal camera lifting:
  https://github.com/HuangJunJie2017/BEVDet
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from tsqbev.reset_stack import component_by_key, recommended_reset_plan


@dataclass(frozen=True, slots=True)
class GapItem:
    """Concrete gap between current and target stack."""

    area: str
    severity: Literal["critical", "high", "medium"]
    current_state: str
    target_state: str
    why_it_matters: str
    recommended_action: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GapReport:
    """Full migration report for the repo reset."""

    current_runtime: str
    target_runtime: str
    move_now_reason: str
    gaps: tuple[GapItem, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "current_runtime": self.current_runtime,
            "target_runtime": self.target_runtime,
            "move_now_reason": self.move_now_reason,
            "gaps": [gap.to_dict() for gap in self.gaps],
        }


def analyze_reset_gap() -> GapReport:
    """Return the evidence-backed gap report for the current repo."""

    plan = recommended_reset_plan()
    return GapReport(
        current_runtime="legacy_sparse_query_multimodal_student",
        target_runtime=plan.name,
        move_now_reason=(
            "The current repo has strong data/eval/teacher plumbing but still depends on custom "
            "query-level detection logic that has underperformed public dense BEV baselines by a "
            "wide margin."
        ),
        gaps=(
            GapItem(
                area="primary_representation",
                severity="critical",
                current_state="Sparse query bank with query-level temporal memory.",
                target_state="Shared dense BEV tensor with short temporal BEV memory.",
                why_it_matters=(
                    "Equal-priority detection and lane/map tasks need one mature shared state, not "
                    "two partially coupled query branches."
                ),
                recommended_action=(
                    "Adopt the BEVFusion/BEVDet representation and keep sparse-query code only as "
                    "a legacy experiment branch."
                ),
            ),
            GapItem(
                area="camera_branch",
                severity="critical",
                current_state="Torchvision backbone plus sparse proposal-ray lifting.",
                target_state=(
                    f"{component_by_key('bevdet4d-bevdepth').name} temporal camera BEV encoder."
                ),
                why_it_matters=(
                    "The repo currently reinvents camera BEV lifting instead of using a stronger, "
                    "publicly validated temporal camera path."
                ),
                recommended_action=(
                    "Replace sparse camera lifting with a BEVDet4D/BEVDepth-compatible encoder."
                ),
            ),
            GapItem(
                area="detection_head",
                severity="critical",
                current_state="Custom query/objectness head with calibration-heavy ranking.",
                target_state=f"{component_by_key('centerhead').name} dense head.",
                why_it_matters=(
                    "Ranking collapse, overproduction, and weak car emergence have all centered on "
                    "the custom head."
                ),
                recommended_action=(
                    "Promote a CenterPoint-style dense heatmap head as the default detector."
                ),
            ),
            GapItem(
                area="multitask_scope",
                severity="high",
                current_state="Detection-first runtime with a light camera-dominant lane branch.",
                target_state=f"{component_by_key('maptrv2').name} on top of shared BEV.",
                why_it_matters=(
                    "Lane/map parity requires the same BEV trunk to support vector outputs, not a "
                    "side branch bolted onto detection queries."
                ),
                recommended_action=(
                    "Move lane/map supervision onto the dense BEV trunk and treat OpenLane as an "
                    "auxiliary eval/teacher path."
                ),
            ),
            GapItem(
                area="teacher_ceiling",
                severity="high",
                current_state="LiDAR-heavy teacher bootstrap used mainly for seed replacement.",
                target_state=(
                    f"{component_by_key('openpcdet-bevfusion').name} or similar multimodal teacher "
                    "with BEV/head distillation."
                ),
                why_it_matters=(
                    "A LiDAR-only teacher cannot teach the camera BEV path or vector-map branch."
                ),
                recommended_action=(
                    "Upgrade the teacher suite to shared-BEV and dense-head distillation targets."
                ),
            ),
            GapItem(
                area="deployment_path",
                severity="high",
                current_state="ONNX/TensorRT exists only for the simplified legacy core.",
                target_state="TensorRT and DeepStream DS3D-compatible dense-BEV deployment path.",
                why_it_matters=(
                    "AGX Orin is a first-class target, so the runtime must align with NVIDIA's "
                    "supported multimodal stack shape."
                ),
                recommended_action=(
                    "Use BEVFusion-compatible TensorRT chunking and DeepStream guidance as the "
                    "deployment contract."
                ),
            ),
            GapItem(
                area="efficiency_specialization",
                severity="medium",
                current_state="Ad hoc backbone selection and local latency heuristics.",
                target_state=f"{component_by_key('ofa-amc-haq').name} after baseline reproduction.",
                why_it_matters=(
                    "The repo needs hardware-aware specialization, but only after the perception "
                    "stack itself is no longer the bottleneck."
                ),
                recommended_action=(
                    "Treat EfficientViT/OFA/AMC/HAQ as phase-2 optimizers, not as a substitute for "
                    "the dense-BEV reset."
                ),
            ),
        ),
    )
