"""Official BEVFusion environment and dataset readiness checks.

Primary sources:
- BEVFusion README:
  https://github.com/mit-han-lab/bevfusion
- BEVFusion docker image recipe:
  https://github.com/mit-han-lab/bevfusion/blob/main/docker/Dockerfile
- BEVFusion nuScenes config contract:
  https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/default.yaml
- Archived official issue on the create_data.py failure mode:
  https://github.com/mit-han-lab/bevfusion/issues/569
- nuScenes devkit releases:
  https://github.com/nutonomy/nuscenes-devkit/releases
- nuScenes devkit:
  https://github.com/nutonomy/nuscenes-devkit
- NVIDIA DeepStream DS3D BEVFusion docs:
  https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_3D_MultiModal_Lidar_Camera_BEVFusion.html
- NVIDIA CUDA-BEVFusion:
  https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BevFusionEnvStatus:
    """Local execution status for the official upstream BEVFusion path."""

    repo_root: str
    dataset_root: str
    docker_available: bool
    docker_version: str | None
    nvidia_ctk_available: bool
    nvidia_container_runtime_available: bool
    host_python: str
    host_python_supported: bool
    dockerfile_present: bool
    download_pretrained_present: bool
    create_data_present: bool
    detection_config_present: bool
    samples_present: bool
    sweeps_present: bool
    maps_present: bool
    map_expansion_present: bool
    map_basemap_present: bool
    map_prediction_present: bool
    trainval_meta_present: bool
    mini_meta_present: bool
    train_info_present: bool
    val_info_present: bool
    blockers: tuple[str, ...]
    notes: tuple[str, ...]
    recommended_strategy: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _command_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            list(args),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _docker_version() -> str | None:
    return _command_output("docker", "--version")


def _host_python_string() -> str:
    return platform.python_version()


def check_bevfusion_environment(repo_root: Path, dataset_root: Path) -> BevFusionEnvStatus:
    """Check whether the workstation can follow the official BEVFusion run path.

    The upstream README pins Python to >=3.8,<3.9 and documents a Dockerfile based
    runtime. This checker treats Docker as the preferred strategy whenever the host
    interpreter falls outside that official range.
    """

    docker_available = shutil.which("docker") is not None
    docker_version = _docker_version() if docker_available else None
    nvidia_ctk_available = shutil.which("nvidia-ctk") is not None
    nvidia_container_runtime_available = shutil.which("nvidia-container-runtime") is not None

    host_python = _host_python_string()
    version_parts = tuple(int(part) for part in host_python.split(".")[:2])
    host_python_supported = (3, 8) <= version_parts < (3, 9)

    dockerfile_present = (repo_root / "docker" / "Dockerfile").exists()
    download_pretrained_present = (repo_root / "tools" / "download_pretrained.sh").exists()
    create_data_present = (repo_root / "tools" / "create_data.py").exists()
    detection_config_present = (
        repo_root
        / "configs"
        / "nuscenes"
        / "det"
        / "transfusion"
        / "secfpn"
        / "camera+lidar"
        / "swint_v0p075"
        / "convfuser.yaml"
    ).exists()

    samples_present = (dataset_root / "samples").exists()
    sweeps_present = (dataset_root / "sweeps").exists()
    maps_present = (dataset_root / "maps").exists()
    map_expansion_present = (dataset_root / "maps" / "expansion").exists()
    map_basemap_present = (dataset_root / "maps" / "basemap").exists()
    map_prediction_present = (dataset_root / "maps" / "prediction").exists()
    trainval_meta_present = (dataset_root / "v1.0-trainval").exists()
    mini_meta_present = (dataset_root / "v1.0-mini").exists()
    train_info_present = (dataset_root / "nuscenes_infos_train.pkl").exists()
    val_info_present = (dataset_root / "nuscenes_infos_val.pkl").exists()

    blockers: list[str] = []
    if not repo_root.exists():
        blockers.append("BEVFusion repo root missing")
    if not dataset_root.exists():
        blockers.append("nuScenes dataset root missing")
    if not docker_available:
        blockers.append("docker missing")
    if not nvidia_container_runtime_available:
        blockers.append("nvidia-container-runtime missing")
    if not dockerfile_present:
        blockers.append("official BEVFusion Dockerfile missing")
    if not create_data_present:
        blockers.append("BEVFusion data prep script missing")
    if not detection_config_present:
        blockers.append("BEVFusion nuScenes detection config missing")
    if not samples_present:
        blockers.append("nuScenes samples directory missing")
    if not sweeps_present:
        blockers.append("nuScenes sweeps directory missing")
    if not trainval_meta_present:
        blockers.append("nuScenes v1.0-trainval metadata missing")
    if not maps_present:
        blockers.append("nuScenes maps directory missing")
    if maps_present and not map_basemap_present:
        blockers.append("nuScenes map expansion missing under maps/basemap")
    if maps_present and not map_expansion_present:
        blockers.append("nuScenes map expansion missing under maps/expansion")
    if maps_present and not map_prediction_present:
        blockers.append("nuScenes map expansion missing under maps/prediction")
    notes: list[str] = []
    if not host_python_supported:
        notes.append("Host Python is outside BEVFusion's official >=3.8,<3.9 range; use Docker.")
    if not mini_meta_present:
        notes.append("v1.0-mini metadata missing; mini-only smoke eval is unavailable.")
    if not train_info_present or not val_info_present:
        notes.append(
            "nuScenes ann files are not ready yet; generate nuscenes_infos_train.pkl and "
            "nuscenes_infos_val.pkl with the repo-local helper instead of relying on upstream "
            "tools/create_data.py."
        )
    if create_data_present:
        notes.append(
            "The official BEVFusion create_data.py path is not the preferred eval prep route "
            "here; the repo-local helper writes the exact ann-file names expected by "
            "configs/nuscenes/default.yaml."
        )
    if maps_present and not map_basemap_present:
        notes.append(
            "nuScenes maps/basemap is missing; the official devkit map expansion bundle "
            "includes it, and BEVFusion's shared nuScenes pipeline instantiates NuScenesMap "
            "even for the detection config."
        )
    if maps_present and not map_expansion_present:
        notes.append(
            "nuScenes maps/expansion is missing; this blocks the official BEVFusion detection "
            "and segmentation reproduction because the shared nuScenes pipeline loads "
            "NuScenesMap in both paths."
        )
    if maps_present and not map_prediction_present:
        notes.append(
            "nuScenes maps/prediction is missing; the official devkit map expansion bundle "
            "includes it, and BEVFusion's shared nuScenes pipeline instantiates NuScenesMap "
            "even for the detection config."
        )
    if (dataset_root / "v1.0-trainval" / "samples").exists():
        notes.append(
            "Dataset root contains duplicated extracted content under v1.0-trainval/; keep "
            "root-path at the dataset root."
        )

    recommended_strategy = (
        "official-docker"
        if docker_available and nvidia_container_runtime_available and dockerfile_present
        else "blocked"
    )

    return BevFusionEnvStatus(
        repo_root=str(repo_root),
        dataset_root=str(dataset_root),
        docker_available=docker_available,
        docker_version=docker_version,
        nvidia_ctk_available=nvidia_ctk_available,
        nvidia_container_runtime_available=nvidia_container_runtime_available,
        host_python=host_python,
        host_python_supported=host_python_supported,
        dockerfile_present=dockerfile_present,
        download_pretrained_present=download_pretrained_present,
        create_data_present=create_data_present,
        detection_config_present=detection_config_present,
        samples_present=samples_present,
        sweeps_present=sweeps_present,
        maps_present=maps_present,
        map_expansion_present=map_expansion_present,
        map_basemap_present=map_basemap_present,
        map_prediction_present=map_prediction_present,
        trainval_meta_present=trainval_meta_present,
        mini_meta_present=mini_meta_present,
        train_info_present=train_info_present,
        val_info_present=val_info_present,
        blockers=tuple(blockers),
        notes=tuple(notes),
        recommended_strategy=recommended_strategy,
    )


def bevfusion_official_commands(
    repo_root: Path,
    dataset_root: Path,
    image_tag: str = "tsqbev-bevfusion-official:latest",
    gpu_count: int = 1,
    tsqbev_repo_root: Path = Path("/home/achbogga/projects/tsqbev-poc"),
) -> dict[str, str]:
    """Return concrete commands for the official BEVFusion Docker workflow."""

    repo_root = repo_root.resolve()
    dataset_root = dataset_root.resolve()
    tsqbev_repo_root = tsqbev_repo_root.resolve()
    build = (
        f"cd {tsqbev_repo_root} && "
        f"BEVFUSION_ROOT={repo_root} IMAGE_TAG={image_tag} "
        f"./research/scripts/bootstrap_bevfusion_official.sh"
    )
    prep = (
        f"cd {tsqbev_repo_root} && "
        f"BEVFUSION_ROOT={repo_root} DATASET_ROOT={dataset_root} IMAGE_TAG={image_tag} "
        f"VERSION=v1.0-trainval MAX_SWEEPS=10 "
        f"./research/scripts/run_bevfusion_nuscenes_prep.sh"
    )
    download = (
        f"cd {tsqbev_repo_root} && "
        f"BEVFUSION_ROOT={repo_root} IMAGE_TAG={image_tag} "
        f"./research/scripts/run_bevfusion_download_pretrained.sh"
    )
    evaluate = (
        f"cd {tsqbev_repo_root} && "
        f"BEVFUSION_ROOT={repo_root} DATASET_ROOT={dataset_root} "
        f"IMAGE_TAG={image_tag} NUM_GPUS={gpu_count} "
        f"./research/scripts/run_bevfusion_nuscenes_eval.sh"
    )
    evaluate_seg = (
        f"cd {tsqbev_repo_root} && "
        f"BEVFUSION_ROOT={repo_root} DATASET_ROOT={dataset_root} "
        f"IMAGE_TAG={image_tag} NUM_GPUS={gpu_count} "
        f"CONFIG_REL=configs/nuscenes/seg/fusion-bev256d2-lss.yaml "
        f"CHECKPOINT_PATH=pretrained/bevfusion-seg.pth EVAL_KIND=map "
        f"./research/scripts/run_bevfusion_nuscenes_eval.sh"
    )
    return {
        "build_image": build,
        "prepare_nuscenes": prep,
        "download_pretrained": download,
        "evaluate_detection": evaluate,
        "evaluate_segmentation": evaluate_seg,
    }


def render_bevfusion_runbook_markdown(
    repo_root: Path,
    dataset_root: Path,
    image_tag: str = "tsqbev-bevfusion-official:latest",
    gpu_count: int = 1,
    tsqbev_repo_root: Path = Path("/home/achbogga/projects/tsqbev-poc"),
) -> str:
    """Render the official BEVFusion baseline path as Markdown."""

    status = check_bevfusion_environment(repo_root=repo_root, dataset_root=dataset_root)
    commands = bevfusion_official_commands(
        repo_root=repo_root,
        dataset_root=dataset_root,
        image_tag=image_tag,
        gpu_count=gpu_count,
        tsqbev_repo_root=tsqbev_repo_root,
    )

    lines = [
        "# BEVFusion Baseline Runbook",
        "",
        "Primary sources:",
        "",
        "- https://github.com/mit-han-lab/bevfusion",
        "- https://github.com/mit-han-lab/bevfusion/blob/main/docker/Dockerfile",
        "- https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/default.yaml",
        "- https://github.com/mit-han-lab/bevfusion/issues/569",
        "- https://github.com/nutonomy/nuscenes-devkit/releases",
        (
            "- https://docs.nvidia.com/metropolis/deepstream/7.1/text/"
            "DS_3D_MultiModal_Lidar_Camera_BEVFusion.html"
        ),
        "",
        "## Current Local Status",
        "",
        f"- repo root: `{status.repo_root}`",
        f"- dataset root: `{status.dataset_root}`",
        f"- docker available: `{status.docker_available}`",
        f"- docker version: `{status.docker_version}`",
        f"- nvidia container runtime available: `{status.nvidia_container_runtime_available}`",
        (
            f"- host python supported by upstream: `{status.host_python_supported}` "
            f"(`{status.host_python}`)"
        ),
        (
            f"- ann files present: train=`{status.train_info_present}`, "
            f"val=`{status.val_info_present}`"
        ),
        (
            f"- map expansion present: `{status.map_expansion_present}`, "
            f"basemap=`{status.map_basemap_present}`, "
            f"prediction=`{status.map_prediction_present}`"
        ),
    ]
    if status.blockers:
        lines.extend(["", "### Blockers", ""])
        lines.extend([f"- {blocker}" for blocker in status.blockers])
    if status.notes:
        lines.extend(["", "### Notes", ""])
        lines.extend([f"- {note}" for note in status.notes])

    lines.extend(
        [
            "",
            "## Build Image",
            "",
            "```bash",
            commands["build_image"],
            "```",
            "",
            "## Prepare nuScenes Ann Files",
            "",
            "```bash",
            commands["prepare_nuscenes"],
            "```",
            "",
            "## Download Official Checkpoints",
            "",
            "```bash",
            commands["download_pretrained"],
            "```",
            "",
            "## Evaluate Detection",
            "",
            "```bash",
            commands["evaluate_detection"],
            "```",
            "",
            "## Evaluate Segmentation",
            "",
            "```bash",
            commands["evaluate_segmentation"],
            "```",
        ]
    )
    return "\n".join(lines)
