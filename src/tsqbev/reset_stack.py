"""Public registry and recommendation logic for the reset stack.

References:
- Sparse4D:
  https://github.com/HorizonRobotics/Sparse4D
- BEVFormer v2:
  https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.pdf
- OpenPCDet:
  https://github.com/open-mmlab/OpenPCDet
- BEVFusion:
  https://github.com/mit-han-lab/bevfusion
- MapTR:
  https://github.com/hustvl/MapTR
- PersFormer:
  https://github.com/OpenDriveLab/PersFormer_3DLane
- DINOv2:
  https://github.com/facebookresearch/dinov2
- DINOv3:
  https://github.com/facebookresearch/dinov3
- EfficientViT:
  https://github.com/mit-han-lab/efficientvit
- NVIDIA Alpamayo:
  https://nvidianews.nvidia.com/news/alpamayo-autonom
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from tsqbev.reset_contracts import BevGridSpec

SourceKind = Literal["paper", "repo", "weights", "docs"]


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Primary-source reference attached to a component or recommendation."""

    label: str
    url: str
    kind: SourceKind


@dataclass(frozen=True, slots=True)
class UpstreamComponent:
    """Curated external component with public code and evidence."""

    key: str
    name: str
    role: str
    modalities: tuple[str, ...]
    tasks: tuple[str, ...]
    repo_url: str
    paper_url: str
    weights_url: str | None
    docs_url: str | None
    code_maturity: Literal["high", "medium", "legacy"]
    deployability: Literal["high", "medium", "low"]
    license_access: str
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlannedComponent:
    """Selected component inside the recommended reset architecture."""

    role: str
    component_key: str
    why_selected: str
    replaces_current: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MigrationMilestone:
    """Concrete migration phase with explicit acceptance criteria."""

    name: str
    objective: str
    acceptance: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResetArchitecturePlan:
    """Decision-complete target stack for the repo reset."""

    name: str
    summary: str
    grid: BevGridSpec
    components: tuple[PlannedComponent, ...]
    milestones: tuple[MigrationMilestone, ...]
    deferred_tracks: tuple[str, ...]
    primary_sources: tuple[SourceRef, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "summary": self.summary,
            "grid": asdict(self.grid),
            "components": [component.to_dict() for component in self.components],
            "milestones": [milestone.to_dict() for milestone in self.milestones],
            "deferred_tracks": list(self.deferred_tracks),
            "primary_sources": [asdict(source) for source in self.primary_sources],
        }


def upstream_registry() -> tuple[UpstreamComponent, ...]:
    """Return the evidence-backed external component registry for the reset stack."""

    return (
        UpstreamComponent(
            key="openpcdet-centerpoint-pointpillar",
            name="OpenPCDet CenterPoint-PointPillar",
            role="runtime_lidar_anchor_prior",
            modalities=("lidar",),
            tasks=("3d_detection",),
            repo_url="https://github.com/open-mmlab/OpenPCDet",
            paper_url=(
                "https://openaccess.thecvf.com/content/CVPR2021/papers/"
                "Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf"
            ),
            weights_url="https://github.com/open-mmlab/OpenPCDet",
            docs_url="https://github.com/open-mmlab/OpenPCDet",
            code_maturity="high",
            deployability="high",
            license_access="Apache-2.0 code; public downloadable nuScenes checkpoints",
            rationale=(
                "Best balance of public weights, mature code, and pillar-based deployability for "
                "runtime LiDAR grounding and anchor priors."
            ),
        ),
        UpstreamComponent(
            key="openpcdet-bevfusion",
            name="OpenPCDet BEVFusion",
            role="upper_bound_multimodal_teacher",
            modalities=("lidar", "camera"),
            tasks=("3d_detection",),
            repo_url="https://github.com/open-mmlab/OpenPCDet",
            paper_url="https://github.com/mit-han-lab/bevfusion",
            weights_url="https://github.com/open-mmlab/OpenPCDet",
            docs_url="https://github.com/open-mmlab/OpenPCDet",
            code_maturity="high",
            deployability="medium",
            license_access="Apache-2.0 code; official model-zoo entries",
            rationale=(
                "Strong public multimodal teacher and realistic control baseline using the same "
                "ecosystem as the LiDAR teacher path."
            ),
        ),
        UpstreamComponent(
            key="mit-bevfusion",
            name="MIT HAN Lab BEVFusion",
            role="control_multimodal_runtime",
            modalities=("lidar", "camera"),
            tasks=("3d_detection", "bev_map"),
            repo_url="https://github.com/mit-han-lab/bevfusion",
            paper_url="https://github.com/mit-han-lab/bevfusion",
            weights_url="https://github.com/mit-han-lab/bevfusion",
            docs_url=(
                "https://docs.nvidia.com/metropolis/deepstream/7.1/text/"
                "DS_3D_MultiModal_Lidar_Camera_BEVFusion.html"
            ),
            code_maturity="high",
            deployability="high",
            license_access="MIT-origin public repo and NVIDIA DeepStream integration docs",
            rationale=(
                "Strong reproduced control baseline and deployment reference, but no longer the "
                "sole runtime thesis."
            ),
        ),
        UpstreamComponent(
            key="sparse4dv3",
            name="Sparse4Dv3",
            role="runtime_sparse_temporal_core",
            modalities=("camera",),
            tasks=("3d_detection", "tracking", "temporal_perception"),
            repo_url="https://github.com/HorizonRobotics/Sparse4D",
            paper_url="https://arxiv.org/abs/2311.11722",
            weights_url="https://github.com/HorizonRobotics/Sparse4D",
            docs_url="https://github.com/HorizonRobotics/Sparse4D",
            code_maturity="high",
            deployability="medium",
            license_access="MIT code with public configs and checkpoints",
            rationale=(
                "Best current public sparse temporal camera-first detection family with strong "
                "nuScenes evidence and a clear runtime story."
            ),
        ),
        UpstreamComponent(
            key="bevformer-v2",
            name="BEVFormer v2",
            role="perspective_supervision_template",
            modalities=("camera",),
            tasks=("3d_detection", "temporal_perception"),
            repo_url="https://github.com/fundamentalvision/BEVFormer",
            paper_url=(
                "https://openaccess.thecvf.com/content/CVPR2023/papers/"
                "Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_"
                "Recognition_via_CVPR_2023_paper.pdf"
            ),
            weights_url="https://github.com/fundamentalvision/BEVFormer",
            docs_url="https://github.com/fundamentalvision/BEVFormer",
            code_maturity="high",
            deployability="medium",
            license_access="Apache-2.0 code and public configs/checkpoints",
            rationale=(
                "Best evidence-backed template for perspective supervision and adapting stronger "
                "image backbones to 3D reasoning."
            ),
        ),
        UpstreamComponent(
            key="maptrv2",
            name="MapTR / MapTRv2",
            role="runtime_vector_map_head",
            modalities=("bev", "camera", "lidar"),
            tasks=("lane_map_vectorization",),
            repo_url="https://github.com/hustvl/MapTR",
            paper_url="https://arxiv.org/abs/2208.14437",
            weights_url="https://github.com/hustvl/MapTR",
            docs_url="https://github.com/hustvl/MapTR",
            code_maturity="high",
            deployability="medium",
            license_access="Public repo with checkpoints and nuScenes map support",
            rationale=(
                "Best fit for staged vector lane/map outputs once the shared latent is stable."
            ),
        ),
        UpstreamComponent(
            key="persformer",
            name="PersFormer",
            role="auxiliary_lane_teacher",
            modalities=("camera",),
            tasks=("lane_detection",),
            repo_url="https://github.com/OpenDriveLab/PersFormer_3DLane",
            paper_url="https://arxiv.org/abs/2203.11089",
            weights_url="https://github.com/OpenDriveLab/PersFormer_3DLane",
            docs_url="https://github.com/OpenDriveLab/PersFormer_3DLane",
            code_maturity="medium",
            deployability="medium",
            license_access="Public repo and benchmark-specific weights",
            rationale=(
                "Useful as an OpenLane-facing lane specialist, but not the shared runtime."
            ),
        ),
        UpstreamComponent(
            key="efficientvit",
            name="EfficientViT",
            role="camera_efficiency_specialization",
            modalities=("camera",),
            tasks=("dense_features", "segmentation"),
            repo_url="https://github.com/mit-han-lab/efficientvit",
            paper_url="https://github.com/mit-han-lab/efficientvit",
            weights_url="https://github.com/mit-han-lab/efficientvit",
            docs_url="https://github.com/mit-han-lab/efficientvit",
            code_maturity="high",
            deployability="high",
            license_access="Apache-2.0 code with pretrained models and deployment recipes",
            rationale=(
                "Best candidate for the student efficiency branch after the perception thesis is "
                "stabilized."
            ),
        ),
        UpstreamComponent(
            key="dinov2",
            name="DINOv2",
            role="dense_feature_teacher",
            modalities=("camera",),
            tasks=("dense_features",),
            repo_url="https://github.com/facebookresearch/dinov2",
            paper_url="https://github.com/facebookresearch/dinov2",
            weights_url="https://github.com/facebookresearch/dinov2",
            docs_url="https://github.com/facebookresearch/dinov2",
            code_maturity="high",
            deployability="medium",
            license_access="Public code and hub-loading backbones",
            rationale=(
                "Low-friction dense visual feature teacher and projector source for the next "
                "camera branch."
            ),
        ),
        UpstreamComponent(
            key="dinov3",
            name="DINOv3",
            role="dense_feature_teacher_next",
            modalities=("camera",),
            tasks=("dense_features",),
            repo_url="https://github.com/facebookresearch/dinov3",
            paper_url="https://github.com/facebookresearch/dinov3",
            weights_url="https://github.com/facebookresearch/dinov3",
            docs_url="https://github.com/facebookresearch/dinov3",
            code_maturity="high",
            deployability="medium",
            license_access="Public code; weight access more involved than DINOv2",
            rationale=(
                "Higher-ceiling dense-feature teacher with public distilled ConvNeXt options for "
                "later camera-side uplift."
            ),
        ),
        UpstreamComponent(
            key="alpamayo-r1",
            name="NVIDIA Alpamayo",
            role="reasoning_teacher_evaluator",
            modalities=("camera", "lidar", "language"),
            tasks=("reasoning", "scenario_mining", "evaluation"),
            repo_url="https://nvidianews.nvidia.com/news/alpamayo-autonom",
            paper_url="https://nvidianews.nvidia.com/news/alpamayo-autonom",
            weights_url=None,
            docs_url="https://nvidianews.nvidia.com/news/alpamayo-autonom",
            code_maturity="medium",
            deployability="low",
            license_access=(
                "Public announcement and ecosystem references; not a local runtime dependency"
            ),
            rationale=(
                "Useful as a teacher-side long-tail reasoner and evaluator, not as the student "
                "perception trunk."
            ),
        ),
        UpstreamComponent(
            key="ofa-amc-haq",
            name="OFA + AMC + HAQ",
            role="hardware_aware_specialization",
            modalities=("meta",),
            tasks=("nas", "pruning", "quantization"),
            repo_url="https://hanlab.mit.edu/projects/ofa",
            paper_url="https://hanlab.mit.edu/projects/ofa",
            weights_url=None,
            docs_url="https://hanlab.mit.edu/",
            code_maturity="high",
            deployability="high",
            license_access="Public research code and project pages from MIT HAN Lab",
            rationale=(
                "The right efficiency toolkit after the perception thesis is proven; not a "
                "replacement for the perception stack itself."
            ),
        ),
    )


def component_by_key(component_key: str) -> UpstreamComponent:
    """Return one component from the registry or raise a clear error."""

    for component in upstream_registry():
        if component.key == component_key:
            return component
    raise KeyError(f"unknown upstream component: {component_key}")


def recommended_reset_plan() -> ResetArchitecturePlan:
    """Return the evidence-backed migration target for the repo."""

    return ResetArchitecturePlan(
        name="foundation-teacher-perspective-sparse-reset",
        summary=(
            "Replace the ad hoc local runtime with a Sparse4D-style sparse temporal student that "
            "uses DINO camera priors, BEVFormer-v2-style perspective supervision, and strong "
            "OpenPCDet/BEVFusion teachers while retaining the repo's teacher/eval/research "
            "discipline."
        ),
        grid=BevGridSpec(
            x_range_m=(-54.0, 54.0),
            y_range_m=(-54.0, 54.0),
            cell_size_m=0.6,
            channels=128,
            temporal_frames=4,
        ),
        components=(
            PlannedComponent(
                role="camera_sparse_runtime",
                component_key="sparse4dv3",
                why_selected=(
                    "Sparse temporal aggregation is now the strongest public runtime thesis that "
                    "beats pure dense-BEV as a camera-first frontier."
                ),
                replaces_current="local sparse query fusion heuristics",
            ),
            PlannedComponent(
                role="camera_perspective_supervision",
                component_key="bevformer-v2",
                why_selected=(
                    "Perspective supervision directly fixes the weak camera-backbone adaptation "
                    "problem highlighted by the literature."
                ),
                replaces_current="BEV-only supervision on the camera branch",
            ),
            PlannedComponent(
                role="camera_foundation_teacher",
                component_key="dinov2",
                why_selected=(
                    "Lowest-friction public foundation feature teacher with direct hub loading."
                ),
                replaces_current="scratch or torchvision-only camera feature bootstrapping",
            ),
            PlannedComponent(
                role="camera_foundation_teacher_next",
                component_key="dinov3",
                why_selected=(
                    "Higher-ceiling dense-feature teacher with public distilled ConvNeXt options."
                ),
                replaces_current="weak camera priors once DINOv2 is exhausted",
            ),
            PlannedComponent(
                role="lidar_anchor_runtime",
                component_key="openpcdet-centerpoint-pointpillar",
                why_selected=(
                    "Pillar-based LiDAR anchors preserve deployability while grounding depth and "
                    "3D box initialization."
                ),
                replaces_current="lightweight LiDAR seed encoder",
            ),
            PlannedComponent(
                role="multimodal_teacher",
                component_key="openpcdet-bevfusion",
                why_selected=(
                    "Use a dense multimodal teacher for geometry and BEV-space supervision without "
                    "forcing the student into the same runtime shape."
                ),
                replaces_current="LiDAR-only teacher bootstrap ceiling",
            ),
            PlannedComponent(
                role="control_runtime",
                component_key="mit-bevfusion",
                why_selected=(
                    "Keep a reproduced dense-BEV control and deployment reference instead of "
                    "pretending it is no longer relevant."
                ),
                replaces_current="unverified dense-BEV assumptions",
            ),
            PlannedComponent(
                role="vector_map_head",
                component_key="maptrv2",
                why_selected=(
                    "Vector-map head gives the right lane/map formulation once the shared latent "
                    "is stable."
                ),
                replaces_current="camera-dominant lane head",
            ),
            PlannedComponent(
                role="reasoning_teacher",
                component_key="alpamayo-r1",
                why_selected=(
                    "Use teacher-side reasoning and long-tail mining without burdening the "
                    "student runtime."
                ),
                replaces_current="ad hoc manual hard-case interpretation",
            ),
            PlannedComponent(
                role="camera_efficiency_track",
                component_key="efficientvit",
                why_selected=(
                    "Add only after the new student branch is real to specialize cost for Orin."
                ),
                replaces_current="ad hoc torchvision backbone selection",
            ),
        ),
        milestones=(
            MigrationMilestone(
                name="reproduce_unrestricted_teachers",
                objective=(
                    "Evaluate official OpenPCDet/BEVFusion/Sparse4D and DINO feature-loading "
                    "paths locally and record the exact ceilings before new training."
                ),
                acceptance=(
                    "All public teachers or control baselines run and land within documented "
                    "tolerance."
                ),
            ),
            MigrationMilestone(
                name="foundation_camera_projector",
                objective=(
                    "Add frozen DINOv2 first and DINOv3 second as projected camera features into "
                    "the runtime student."
                ),
                acceptance=(
                    "The projected-feature student trains stably and beats the non-foundation "
                    "camera branch on official mini metrics."
                ),
            ),
            MigrationMilestone(
                name="perspective_supervised_sparse_student",
                objective=(
                    "Stand up the first repo-native student that combines sparse temporal "
                    "aggregation, perspective supervision, and LiDAR anchors."
                ),
                acceptance=(
                    "The new student materially beats the current local frontier on official mini "
                    "metrics."
                ),
            ),
            MigrationMilestone(
                name="teacher_driven_bev_and_quality_lift",
                objective=(
                    "Distill quality maps, proposal quality, and BEV-space teacher signals from "
                    "the unrestricted teachers."
                ),
                acceptance=(
                    "Teacher-on runs materially beat teacher-off runs on official mini metrics."
                ),
            ),
            MigrationMilestone(
                name="lane_and_efficiency_specialization",
                objective=(
                    "Add the MapTRv2 lane/vector head under detection non-regression, then apply "
                    "HAN-Lab style specialization and TensorRT-aware quantization."
                ),
                acceptance=(
                    "The student has a credible AGX Orin FP16/INT8 path and a measured joint "
                    "detection+lane non-regression artifact."
                ),
            ),
        ),
        deferred_tracks=(
            "Pure dense-BEV runtime migration as the main architecture thesis",
            "SAM2 offline proposal priors after detection and vector-map quality are real",
            "Any further weight-only local winner-line nudges after two stalled runs",
        ),
        primary_sources=(
            SourceRef("Sparse4D", "https://github.com/HorizonRobotics/Sparse4D", "repo"),
            SourceRef(
                "BEVFormer v2",
                (
                    "https://openaccess.thecvf.com/content/CVPR2023/papers/"
                    "Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_"
                    "Recognition_via_CVPR_2023_paper.pdf"
                ),
                "paper",
            ),
            SourceRef("OpenPCDet", "https://github.com/open-mmlab/OpenPCDet", "repo"),
            SourceRef("MapTR", "https://github.com/hustvl/MapTR", "repo"),
            SourceRef("DINOv2", "https://github.com/facebookresearch/dinov2", "repo"),
            SourceRef("DINOv3", "https://github.com/facebookresearch/dinov3", "repo"),
            SourceRef(
                "NVIDIA Alpamayo",
                "https://nvidianews.nvidia.com/news/alpamayo-autonom",
                "docs",
            ),
        ),
    )


def render_reset_plan_markdown() -> str:
    """Render the reset recommendation as a compact Markdown report."""

    plan = recommended_reset_plan()
    lines = [f"# {plan.name}", "", plan.summary, "", "## Components"]
    for component in plan.components:
        selected = component_by_key(component.component_key)
        lines.append(
            f"- `{component.role}`: **{selected.name}**. {component.why_selected} "
            f"Replaces `{component.replaces_current}`."
        )
    lines.append("")
    lines.append("## Milestones")
    for milestone in plan.milestones:
        lines.append(
            f"- **{milestone.name}**: {milestone.objective} Acceptance: {milestone.acceptance}"
        )
    lines.append("")
    lines.append("## Deferred Tracks")
    for track in plan.deferred_tracks:
        lines.append(f"- {track}")
    return "\n".join(lines)
