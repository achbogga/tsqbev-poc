"""Gap analysis between the legacy sparse-query repo and the reset target.

References:
- Sparse4D:
  https://github.com/HorizonRobotics/Sparse4D
- BEVFormer v2:
  https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.pdf
- OpenPCDet:
  https://github.com/open-mmlab/OpenPCDet
- DINOv2:
  https://github.com/facebookresearch/dinov2
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
            "The current repo has strong data/eval/teacher plumbing but still lacks a runtime "
            "camera path that exploits foundation features, perspective supervision, and stronger "
            "teacher outputs."
        ),
        gaps=(
            GapItem(
                area="primary_representation",
                severity="critical",
                current_state="Sparse query bank with query-level temporal memory.",
                target_state=(
                    "Sparse temporal instance memory with perspective-supervised camera features "
                    "and a compact shared latent for lane/map supervision."
                ),
                why_it_matters=(
                    "The current runtime has neither the sparse temporal maturity of Sparse4D nor "
                    "the direct backbone supervision of BEVFormer v2."
                ),
                recommended_action=(
                    "Promote a Sparse4D-style temporal core and keep the old local query stack "
                    "only as legacy evidence."
                ),
            ),
            GapItem(
                area="camera_foundation",
                severity="critical",
                current_state="Torchvision backbone plus sparse proposal-ray lifting.",
                target_state=(
                    f"{component_by_key('dinov2').name} / {component_by_key('dinov3').name} "
                    "projected into the camera branch."
                ),
                why_it_matters=(
                    "The repo is still trying to learn high-level camera features from scratch on "
                    "tiny datasets instead of reusing public foundation features."
                ),
                recommended_action=(
                    "Add a frozen camera foundation projector path before opening more local "
                    "runtime tweaks."
                ),
            ),
            GapItem(
                area="perspective_supervision",
                severity="critical",
                current_state="BEV-only supervision and ranking-heavy teacher guidance.",
                target_state=(
                    f"{component_by_key('bevformer-v2').name} style perspective auxiliary head."
                ),
                why_it_matters=(
                    "Public evidence says strong image backbones need direct perspective-side 3D "
                    "supervision to adapt well to BEV or sparse 3D detection."
                ),
                recommended_action=(
                    "Add a perspective head before spending more budget on ranking-only fixes."
                ),
            ),
            GapItem(
                area="teacher_suite",
                severity="high",
                current_state="LiDAR-heavy teacher bootstrap used mainly for seed replacement.",
                target_state=(
                    f"{component_by_key('openpcdet-bevfusion').name} plus "
                    f"{component_by_key('sparse4dv3').name} / foundation camera teachers."
                ),
                why_it_matters=(
                    "A geometry-only teacher cannot teach the camera branch to represent 3D cues "
                    "the way stronger camera-temporal systems do."
                ),
                recommended_action=(
                    "Upgrade the teacher suite to geometry, camera-foundation, and sparse-camera "
                    "teachers rather than only stronger anchor replacement."
                ),
            ),
            GapItem(
                area="lane_multitask",
                severity="high",
                current_state="Detection-first runtime with a light camera-dominant lane branch.",
                target_state=f"{component_by_key('maptrv2').name} on top of the shared latent.",
                why_it_matters=(
                    "Lane/map parity needs a real vector head and explicit non-regression gates, "
                    "not an afterthought side branch."
                ),
                recommended_action=(
                    "Stage lane separately on OpenLane first, then add MapTRv2-style supervision "
                    "only after the latent is stable."
                ),
            ),
            GapItem(
                area="deployment_path",
                severity="high",
                current_state="ONNX/TensorRT exists mainly for simplified legacy or control paths.",
                target_state=(
                    "Activation-checkpointed, GPU-auto-fit student with a credible Orin "
                    "deployment path."
                ),
                why_it_matters=(
                    "The next student branch will use stronger teachers and camera features, so it "
                    "must adapt to local GPU RAM without becoming a one-off lab artifact."
                ),
                recommended_action=(
                    "Add activation checkpointing, GPU auto-fit, and keep BEVFusion/DeepStream as "
                    "control deployment references while the new student matures."
                ),
            ),
        ),
    )
