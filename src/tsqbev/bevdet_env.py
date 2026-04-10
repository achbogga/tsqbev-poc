"""Official BEVDet environment checks, runbooks, and bounded public-student runner.

Primary sources:
- BEVDet README:
  https://github.com/HuangJunJie2017/BEVDet
- BEVDet nuScenes dataset guide:
  https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/docs/en/datasets/nuscenes_det.md
- BEVDet data-prep helper:
  https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/tools/create_data_bevdet.py
- BEVDet Dockerfile:
  https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/docker/Dockerfile
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tsqbev.eval_nuscenes import evaluate_nuscenes_predictions, prediction_geometry_diagnostics

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEVDET_IMAGE_TAG = "tsqbev-bevdet-official:latest"
PRIMARY_BEVDET_CONFIG = "configs/bevdet/bevdet-r50-cbgs.py"
CONTROL_BEVDET_CONFIG = "configs/bevdet/bevdet-r50-4d-depth-cbgs.py"
DEFAULT_PRIMARY_CONFIG_KEY = "bevdet-r50-cbgs"
DEFAULT_CONTROL_CONFIG_KEY = "bevdet-r50-4d-depth-cbgs"


@dataclass(frozen=True, slots=True)
class BevDetEnvStatus:
    """Local execution status for the official upstream BEVDet path."""

    repo_root: str
    dataset_root: str
    docker_available: bool
    docker_version: str | None
    dockerfile_present: bool
    create_data_present: bool
    primary_config_present: bool
    control_config_present: bool
    samples_present: bool
    sweeps_present: bool
    maps_present: bool
    trainval_meta_present: bool
    mini_meta_present: bool
    train_info_present: bool
    val_info_present: bool
    mini_train_info_present: bool
    mini_val_info_present: bool
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


def _docker_image_present(image_tag: str) -> bool:
    completed = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _mini_extra_tag(version: str) -> str:
    return "bevdetv3-mini" if version == "v1.0-mini" else "bevdetv3-nuscenes"


def _reported_latency(config_relpath: str) -> dict[str, Any]:
    if config_relpath == PRIMARY_BEVDET_CONFIG:
        return {
            "mean_ms": 33.2,
            "source": "BEVDet README total latency for R50-CBGS on RTX 3090",
            "fps": 30.1,
        }
    if config_relpath == CONTROL_BEVDET_CONFIG:
        return {
            "mean_ms": 39.7,
            "source": "BEVDet README total latency for R50-4D-Depth-CBGS on RTX 3090",
            "fps": 25.2,
        }
    return {"mean_ms": float("inf"), "source": "no official latency attached"}


def check_bevdet_environment(repo_root: Path, dataset_root: Path) -> BevDetEnvStatus:
    """Check whether the workstation can follow the official BEVDet public-student path."""

    docker_available = shutil.which("docker") is not None
    docker_version = _command_output("docker", "--version") if docker_available else None
    dockerfile_present = (repo_root / "docker" / "Dockerfile").exists()
    create_data_present = (repo_root / "tools" / "create_data_bevdet.py").exists()
    primary_config_present = (repo_root / PRIMARY_BEVDET_CONFIG).exists()
    control_config_present = (repo_root / CONTROL_BEVDET_CONFIG).exists()

    samples_present = (dataset_root / "samples").exists()
    sweeps_present = (dataset_root / "sweeps").exists()
    maps_present = (dataset_root / "maps").exists()
    trainval_meta_present = (dataset_root / "v1.0-trainval").exists()
    mini_meta_present = (dataset_root / "v1.0-mini").exists()
    train_info_present = (dataset_root / "bevdetv3-nuscenes_infos_train.pkl").exists()
    val_info_present = (dataset_root / "bevdetv3-nuscenes_infos_val.pkl").exists()
    mini_train_info_present = (dataset_root / "bevdetv3-mini_infos_train.pkl").exists()
    mini_val_info_present = (dataset_root / "bevdetv3-mini_infos_val.pkl").exists()

    blockers: list[str] = []
    if not repo_root.exists():
        blockers.append("BEVDet repo root missing")
    if not dataset_root.exists():
        blockers.append("nuScenes dataset root missing")
    if not docker_available:
        blockers.append("docker missing")
    if not dockerfile_present:
        blockers.append("official BEVDet Dockerfile missing")
    if not create_data_present:
        blockers.append("BEVDet data prep script missing")
    if not primary_config_present:
        blockers.append("primary BEVDet deployment baseline config missing")
    if not samples_present:
        blockers.append("nuScenes samples directory missing")
    if not sweeps_present:
        blockers.append("nuScenes sweeps directory missing")
    if not trainval_meta_present:
        blockers.append("nuScenes v1.0-trainval metadata missing")
    if not maps_present:
        blockers.append("nuScenes maps directory missing")

    notes: list[str] = []
    if not train_info_present or not val_info_present:
        notes.append(
            "trainval BEVDet info PKLs are missing; generate them with "
            "research/scripts/prepare_bevdet_nuscenes_infos.py."
        )
    if not mini_train_info_present or not mini_val_info_present:
        notes.append(
            "mini BEVDet info PKLs are missing; generate them with "
            "research/scripts/prepare_bevdet_nuscenes_infos.py --version v1.0-mini."
        )
    notes.append(
        "The public-student replacement should start from BEVDet R50-CBGS as the working "
        "deployment baseline and keep R50-4D-Depth-CBGS as the first temporal depth upgrade."
    )
    notes.append(
        "Use the official BEVDet box coder/decoder and postprocess unchanged; do not graft "
        "new backbones onto the custom TSQBEV head."
    )

    recommended_strategy = (
        "official-docker" if docker_available and dockerfile_present else "blocked"
    )
    return BevDetEnvStatus(
        repo_root=str(repo_root),
        dataset_root=str(dataset_root),
        docker_available=docker_available,
        docker_version=docker_version,
        dockerfile_present=dockerfile_present,
        create_data_present=create_data_present,
        primary_config_present=primary_config_present,
        control_config_present=control_config_present,
        samples_present=samples_present,
        sweeps_present=sweeps_present,
        maps_present=maps_present,
        trainval_meta_present=trainval_meta_present,
        mini_meta_present=mini_meta_present,
        train_info_present=train_info_present,
        val_info_present=val_info_present,
        mini_train_info_present=mini_train_info_present,
        mini_val_info_present=mini_val_info_present,
        blockers=tuple(blockers),
        notes=tuple(notes),
        recommended_strategy=recommended_strategy,
    )


def bevdet_official_commands(
    repo_root: Path,
    dataset_root: Path,
    *,
    image_tag: str = DEFAULT_BEVDET_IMAGE_TAG,
    config_relpath: str = PRIMARY_BEVDET_CONFIG,
    epochs: int = 6,
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 2,
    artifact_dir: Path | None = None,
) -> dict[str, str]:
    """Return concrete commands for the official BEVDet Docker workflow."""

    repo_root = repo_root.resolve()
    dataset_root = dataset_root.resolve()
    artifact_dir = (
        artifact_dir.resolve()
        if artifact_dir is not None
        else (REPO_ROOT / "artifacts" / "public_student_bevdet").resolve()
    )
    return {
        "build_image": (
            f"BEVDET_ROOT={repo_root} BEVDET_IMAGE_TAG={image_tag} "
            f"bash {REPO_ROOT}/research/scripts/bootstrap_bevdet_official.sh"
        ),
        "prepare_mini_infos": (
            f"BEVDET_ROOT={repo_root} DATASET_ROOT={dataset_root} VERSION=v1.0-mini "
            f"bash {REPO_ROOT}/research/scripts/run_bevdet_nuscenes_prep.sh"
        ),
        "prepare_trainval_infos": (
            f"BEVDET_ROOT={repo_root} DATASET_ROOT={dataset_root} VERSION=v1.0-trainval "
            f"bash {REPO_ROOT}/research/scripts/run_bevdet_nuscenes_prep.sh"
        ),
        "run_primary_public_student": (
            f"BEVDET_ROOT={repo_root} DATASET_ROOT={dataset_root} "
            f"BEVDET_IMAGE_TAG={image_tag} CONFIG_REL={config_relpath} "
            f"EPOCHS={epochs} SAMPLES_PER_GPU={samples_per_gpu} WORKERS_PER_GPU={workers_per_gpu} "
            f"ARTIFACT_DIR={artifact_dir} VERSION=v1.0-mini "
            f"bash {REPO_ROOT}/research/scripts/run_bevdet_nuscenes_train.sh"
        ),
        "run_control_public_student": (
            f"BEVDET_ROOT={repo_root} DATASET_ROOT={dataset_root} "
            f"BEVDET_IMAGE_TAG={image_tag} CONFIG_REL={CONTROL_BEVDET_CONFIG} "
            f"EPOCHS={epochs} SAMPLES_PER_GPU={samples_per_gpu} WORKERS_PER_GPU={workers_per_gpu} "
            f"ARTIFACT_DIR={artifact_dir} VERSION=v1.0-mini "
            f"bash {REPO_ROOT}/research/scripts/run_bevdet_nuscenes_train.sh"
        ),
    }


def render_bevdet_runbook_markdown(
    repo_root: Path,
    dataset_root: Path,
    *,
    image_tag: str = DEFAULT_BEVDET_IMAGE_TAG,
    config_relpath: str = PRIMARY_BEVDET_CONFIG,
    epochs: int = 6,
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 2,
) -> str:
    """Render the official BEVDet baseline path as Markdown."""

    status = check_bevdet_environment(repo_root=repo_root, dataset_root=dataset_root)
    commands = bevdet_official_commands(
        repo_root=repo_root,
        dataset_root=dataset_root,
        image_tag=image_tag,
        config_relpath=config_relpath,
        epochs=epochs,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )
    lines = [
        "# BEVDet Public Student Runbook",
        "",
        "Primary sources:",
        "- https://github.com/HuangJunJie2017/BEVDet",
        "- https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/docs/en/datasets/nuscenes_det.md",
        "- https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/tools/create_data_bevdet.py",
        "- https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/docker/Dockerfile",
        "",
        "## Status",
    ]
    lines.extend(
        [
            f"- `{key}`: `{value}`"
            for key, value in status.to_dict().items()
            if key not in {"notes", "blockers"}
        ]
    )
    lines.extend(["", "## Blockers"])
    if status.blockers:
        lines.extend([f"- {blocker}" for blocker in status.blockers])
    else:
        lines.append("- none")
    lines.extend(["", "## Notes"])
    lines.extend([f"- {note}" for note in status.notes])
    lines.extend(["", "## Commands"])
    for key, command in commands.items():
        lines.extend([f"### {key}", "```bash", command, "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def run_bevdet_public_student(
    *,
    repo_root: Path,
    dataset_root: Path,
    artifact_dir: Path,
    config_relpath: str,
    version: str,
    epochs: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
    image_tag: str = DEFAULT_BEVDET_IMAGE_TAG,
    load_from: Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Run one bounded public BEVDet student experiment with repo-side evaluation."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    extra_tag = _mini_extra_tag(version)
    split_name = split or ("mini_val" if version == "v1.0-mini" else "val")
    train_info = dataset_root / f"{extra_tag}_infos_train.pkl"
    val_info = dataset_root / f"{extra_tag}_infos_val.pkl"

    if not train_info.exists() or not val_info.exists():
        prep_env = {
            **os.environ,
            "BEVDET_ROOT": str(repo_root),
            "DATASET_ROOT": str(dataset_root),
            "VERSION": version,
        }
        subprocess.run(
            ["bash", str(REPO_ROOT / "research" / "scripts" / "run_bevdet_nuscenes_prep.sh")],
            cwd=REPO_ROOT,
            env=prep_env,
            check=True,
        )

    if not _docker_image_present(image_tag):
        build_env = {
            **os.environ,
            "BEVDET_ROOT": str(repo_root),
            "BEVDET_IMAGE_TAG": image_tag,
        }
        subprocess.run(
            ["bash", str(REPO_ROOT / "research" / "scripts" / "bootstrap_bevdet_official.sh")],
            cwd=REPO_ROOT,
            env=build_env,
            check=True,
        )

    summary_path = artifact_dir / "bevdet_run_summary.json"
    run_env = {
        **os.environ,
        "BEVDET_ROOT": str(repo_root),
        "DATASET_ROOT": str(dataset_root),
        "BEVDET_IMAGE_TAG": image_tag,
        "CONFIG_REL": config_relpath,
        "VERSION": version,
        "ARTIFACT_DIR": str(artifact_dir),
        "EPOCHS": str(epochs),
        "SAMPLES_PER_GPU": str(samples_per_gpu),
        "WORKERS_PER_GPU": str(workers_per_gpu),
        "SUMMARY_PATH": str(summary_path),
    }
    if load_from is not None:
        run_env["LOAD_FROM"] = str(load_from)
    subprocess.run(
        ["bash", str(REPO_ROOT / "research" / "scripts" / "run_bevdet_nuscenes_train.sh")],
        cwd=REPO_ROOT,
        env=run_env,
        check=True,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prediction_path = Path(str(summary["prediction_path"]))
    evaluation = evaluate_nuscenes_predictions(
        dataroot=dataset_root,
        version=version,
        split=split_name,
        result_path=prediction_path,
        output_dir=artifact_dir / "official_eval" / split_name,
    )
    prediction_geometry = prediction_geometry_diagnostics(
        prediction_path,
        dataroot=dataset_root,
        version=version,
    )
    return {
        "backend": "bevdet_official",
        "repo_root": str(repo_root),
        "image_tag": image_tag,
        "config_relpath": config_relpath,
        "checkpoint_path": str(summary["checkpoint_path"]),
        "prediction_path": str(prediction_path),
        "evaluation": evaluation,
        "prediction_geometry": prediction_geometry,
        "benchmark": _reported_latency(config_relpath),
        "durations_s": summary.get("durations_s", {}),
        "train_log": summary.get("train_log"),
        "test_log": summary.get("test_log"),
        "extra_tag": extra_tag,
        "version": version,
        "split": split_name,
    }
