"""Check local prerequisites for the external OpenPCDet teacher workflow.

References:
- OpenPCDet install guide:
  https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md
- OpenPCDet getting started guide:
  https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _find_executable(name: str) -> str | None:
    return shutil.which(name)


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", "installed")


def _read_torch_cuda_home() -> str | None:
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        return None
    return str(CUDA_HOME) if CUDA_HOME is not None else None


def check_openpcdet_environment(repo_root: str | Path) -> dict[str, Any]:
    """Return a machine-readable readiness summary for OpenPCDet teacher bootstrap."""

    repo_root = Path(repo_root)
    install_doc = repo_root / "docs" / "INSTALL.md"
    config_path = repo_root / "tools" / "cfgs" / "nuscenes_models" / "cbgs_dyn_pp_centerpoint.yaml"
    nvcc_path = _find_executable("nvcc")
    torch_version = _module_version("torch")
    torchvision_version = _module_version("torchvision")
    spconv_version = _module_version("spconv")
    torch_scatter_version = _module_version("torch_scatter")
    cuda_home_env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    torch_cuda_home = _read_torch_cuda_home()

    ready = all(
        (
            repo_root.exists(),
            install_doc.exists(),
            config_path.exists(),
            nvcc_path is not None,
            torch_version is not None,
            spconv_version is not None,
            torch_scatter_version is not None,
            (cuda_home_env is not None) or (torch_cuda_home is not None),
        )
    )

    blockers: list[str] = []
    if not repo_root.exists():
        blockers.append("OpenPCDet repo root not found")
    if not install_doc.exists():
        blockers.append("OpenPCDet install guide missing under repo root")
    if not config_path.exists():
        blockers.append("CenterPoint-PointPillar config missing")
    if torch_version is None:
        blockers.append("torch is not installed in the active Python environment")
    if nvcc_path is None:
        blockers.append("nvcc is not on PATH")
    if cuda_home_env is None and torch_cuda_home is None:
        blockers.append("CUDA_HOME/CUDA_PATH is unset and torch reports CUDA_HOME=None")
    if spconv_version is None:
        blockers.append("spconv is not installed in the active Python environment")
    if torch_scatter_version is None:
        blockers.append("torch_scatter is not installed in the active Python environment")

    nvidia_smi: str | None = None
    if _find_executable("nvidia-smi") is not None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            nvidia_smi = completed.stdout.strip()
        except Exception:
            nvidia_smi = None

    return {
        "status": "ready" if ready else "blocked",
        "repo_root": str(repo_root),
        "python_version": sys.version.split()[0],
        "torch": torch_version,
        "torchvision": torchvision_version,
        "spconv": spconv_version,
        "torch_scatter": torch_scatter_version,
        "nvcc_path": nvcc_path,
        "cuda_home_env": cuda_home_env,
        "torch_cuda_home": torch_cuda_home,
        "nvidia_smi": nvidia_smi,
        "install_doc": str(install_doc),
        "config_path": str(config_path),
        "blockers": blockers,
    }
