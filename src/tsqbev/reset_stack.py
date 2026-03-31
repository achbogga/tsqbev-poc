"""Public registry and recommendation logic for the dense-BEV reset stack.

References:
- MIT HAN Lab Once-for-All:
  https://hanlab.mit.edu/projects/ofa
- MIT HAN Lab EfficientViT:
  https://github.com/mit-han-lab/efficientvit
- OpenPCDet model zoo:
  https://github.com/open-mmlab/OpenPCDet
- BEVFusion:
  https://github.com/mit-han-lab/bevfusion
- BEVDet:
  https://github.com/HuangJunJie2017/BEVDet
- MapTR:
  https://github.com/hustvl/MapTR
- PersFormer:
  https://github.com/OpenDriveLab/PersFormer_3DLane
- DINOv2:
  https://github.com/facebookresearch/dinov2
- DINOv3:
  https://github.com/facebookresearch/dinov3
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
            role="runtime_lidar_branch",
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
                "the runtime LiDAR branch."
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
                "Strong public multimodal teacher and realistic integration baseline using "
                "the same ecosystem as the LiDAR branch."
            ),
        ),
        UpstreamComponent(
            key="bevdet4d-bevdepth",
            name="BEVDet4D / BEVDepth camera BEV encoder",
            role="runtime_camera_branch",
            modalities=("camera",),
            tasks=("3d_detection", "temporal_bev"),
            repo_url="https://github.com/HuangJunJie2017/BEVDet",
            paper_url="https://github.com/HuangJunJie2017/BEVDet",
            weights_url="https://github.com/HuangJunJie2017/BEVDet",
            docs_url="https://github.com/HuangJunJie2017/BEVDet",
            code_maturity="high",
            deployability="high",
            license_access="Public repo with published model table and backend latency data",
            rationale=(
                "Most practical public camera-only temporal BEV lift path with explicit TensorRT "
                "latency evidence."
            ),
        ),
        UpstreamComponent(
            key="mit-bevfusion",
            name="MIT HAN Lab BEVFusion",
            role="shared_bev_fusion_trunk",
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
                "Only public stack in the current evidence set that already unifies detection and "
                "BEV map segmentation with official NVIDIA deployment guidance."
            ),
        ),
        UpstreamComponent(
            key="centerhead",
            name="CenterPoint CenterHead",
            role="runtime_detection_head",
            modalities=("bev",),
            tasks=("3d_detection",),
            repo_url="https://github.com/open-mmlab/OpenPCDet",
            paper_url=(
                "https://openaccess.thecvf.com/content/CVPR2021/papers/"
                "Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf"
            ),
            weights_url=None,
            docs_url="https://github.com/open-mmlab/OpenPCDet",
            code_maturity="high",
            deployability="high",
            license_access="Comes from the public OpenPCDet/CenterPoint stack",
            rationale=(
                "Dense heatmap-based head removes the current repo's ranking and query-selection "
                "failure modes."
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
                "Best fit for equal-priority vector map and lane outputs on a shared BEV trunk."
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
                "Useful as an OpenLane-facing lane specialist, but not the shared "
                "multimodal runtime."
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
                "Best candidate for the camera efficiency branch after baseline reproduction."
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
                "Low-friction dense visual feature teacher for later BEV feature distillation."
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
            license_access="Public code; weight access currently more cumbersome than DINOv2",
            rationale=(
                "Stronger dense-feature frontier option, but not the first runtime "
                "backbone for Orin."
            ),
        ),
        UpstreamComponent(
            key="sam2",
            name="SAM2",
            role="offline_proposal_prior",
            modalities=("camera",),
            tasks=("segmentation", "proposal_priors"),
            repo_url="https://github.com/facebookresearch/sam2",
            paper_url="https://github.com/facebookresearch/sam2",
            weights_url="https://github.com/facebookresearch/sam2",
            docs_url="https://github.com/facebookresearch/sam2",
            code_maturity="high",
            deployability="low",
            license_access=(
                "Public code and models; better as offline prior than runtime "
                "encoder here"
            ),
            rationale=(
                "Useful for later proposal priors, not as the primary multi-view "
                "runtime encoder."
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
                "The right efficiency toolkit after the reset baseline exists; not a "
                "replacement for "
                "the perception stack itself."
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
        name="dense-bev-multitask-reset",
        summary=(
            "Replace the custom sparse-query runtime with a dense BEV fusion stack built from "
            "OpenPCDet, BEVDet/BEVDepth, BEVFusion, and MapTR, while retaining the repo's "
            "teacher/eval/research discipline."
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
                role="lidar_runtime",
                component_key="openpcdet-centerpoint-pointpillar",
                why_selected=(
                    "Pillar-based LiDAR path preserves deployability and public checkpoint quality."
                ),
                replaces_current="lightweight LiDAR seed encoder",
            ),
            PlannedComponent(
                role="camera_runtime",
                component_key="bevdet4d-bevdepth",
                why_selected=(
                    "Temporal camera BEV lift is a solved upstream problem and should "
                    "not be rebuilt "
                    "around sparse per-query image sampling."
                ),
                replaces_current="sparse proposal-ray camera path",
            ),
            PlannedComponent(
                role="fusion_trunk",
                component_key="mit-bevfusion",
                why_selected=(
                    "Single dense BEV tensor can serve both detection and vector-map heads."
                ),
                replaces_current="query-level multimodal fusion blocks",
            ),
            PlannedComponent(
                role="detection_head",
                component_key="centerhead",
                why_selected=(
                    "Dense head removes ranking collapse and overproduction failure modes."
                ),
                replaces_current="custom objectness/query head",
            ),
            PlannedComponent(
                role="vector_map_head",
                component_key="maptrv2",
                why_selected=(
                    "Vector-map head matches equal-priority lane/map scope better than the current "
                    "light camera-dominant branch."
                ),
                replaces_current="camera-dominant lane head",
            ),
            PlannedComponent(
                role="teacher_upper_bound",
                component_key="openpcdet-bevfusion",
                why_selected=(
                    "Use a stronger public multimodal teacher before inventing new KD targets."
                ),
                replaces_current="LiDAR-only teacher bootstrap ceiling",
            ),
            PlannedComponent(
                role="camera_efficiency_track",
                component_key="efficientvit",
                why_selected=(
                    "Add only after baseline reproduction to specialize camera cost for Orin."
                ),
                replaces_current="ad hoc torchvision backbone selection",
            ),
        ),
        milestones=(
            MigrationMilestone(
                name="reproduce_upstream_baselines",
                objective=(
                    "Evaluate official OpenPCDet/BEVFusion/BEVDet/MapTR checkpoints locally and "
                    "record the exact ceilings before new training."
                ),
                acceptance="All public baselines run and land within documented tolerance.",
            ),
            MigrationMilestone(
                name="integrated_dense_bev_student",
                objective=(
                    "Stand up the first repo-native dense-BEV student using upstream-compatible "
                    "interfaces and the shared BEV tensor."
                ),
                acceptance=(
                    "Single integrated stack yields official detection and vector-map outputs from "
                    "the same BEV trunk."
                ),
            ),
            MigrationMilestone(
                name="teacher_driven_lift",
                objective=(
                    "Distill BEV features and dense heads from strong public teachers into the "
                    "production student."
                ),
                acceptance="Teacher-on runs beat teacher-off runs on official mini metrics.",
            ),
            MigrationMilestone(
                name="hardware_specialization",
                objective=(
                    "Use HAN-Lab style specialization and TensorRT-aware quantization for the "
                    "production student."
                ),
                acceptance="The student has a credible AGX Orin FP16/INT8 deployment path.",
            ),
        ),
        deferred_tracks=(
            "DINOv2/DINOv3 dense-feature teachers after the BEV baseline exists",
            "SAM2 offline proposal priors after detection and vector-map quality are real",
            "Any further query-only runtime research until the dense-BEV reset is proven",
        ),
        primary_sources=(
            SourceRef("BEVFusion", "https://github.com/mit-han-lab/bevfusion", "repo"),
            SourceRef("OpenPCDet", "https://github.com/open-mmlab/OpenPCDet", "repo"),
            SourceRef("BEVDet", "https://github.com/HuangJunJie2017/BEVDet", "repo"),
            SourceRef("MapTR", "https://github.com/hustvl/MapTR", "repo"),
            SourceRef(
                "DeepStream DS3D BEVFusion",
                (
                    "https://docs.nvidia.com/metropolis/deepstream/7.1/text/"
                    "DS_3D_MultiModal_Lidar_Camera_BEVFusion.html"
                ),
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
