#!/usr/bin/env bash
set -euo pipefail

# Build the official BEVFusion image through a tiny wrapper Dockerfile.
#
# Primary sources:
# - Official BEVFusion Dockerfile:
#   https://github.com/mit-han-lab/bevfusion/blob/main/docker/Dockerfile
# - Anaconda Miniconda installer index:
#   https://repo.anaconda.com/miniconda/
# - MMCV installation guide for pre-built wheels:
#   https://mmcv.readthedocs.io/en/v1.4.6/get_started/installation.html
# - Local build failures on 2026-03-31 using Miniconda latest plus newer channel policy.

BEVFUSION_ROOT="${BEVFUSION_ROOT:-/home/achbogga/projects/bevfusion}"
IMAGE_TAG="${IMAGE_TAG:-tsqbev-bevfusion-official:latest}"

if [[ ! -f "${BEVFUSION_ROOT}/docker/Dockerfile" ]]; then
  echo "missing Dockerfile at ${BEVFUSION_ROOT}/docker/Dockerfile" >&2
  exit 1
fi

TMP_DOCKERFILE="$(mktemp /tmp/bevfusion-dockerfile.XXXXXX)"
cleanup() {
  rm -f "${TMP_DOCKERFILE}"
}
trap cleanup EXIT

python3 - "${BEVFUSION_ROOT}/docker/Dockerfile" "${TMP_DOCKERFILE}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
target = Path(sys.argv[2])

injected = []
for line in source:
    line = line.replace(
        "Miniconda3-latest-Linux-x86_64.sh",
        "Miniconda3-py38_4.12.0-Linux-x86_64.sh",
    )
    if line.strip() == "ENV PATH=$CONDA_DIR/bin:$PATH":
        injected.append(line)
        injected.append("RUN conda create -y -n bevfusion python=3.8")
        injected.append("ENV PATH=$CONDA_DIR/envs/bevfusion/bin:$PATH")
        continue
    if line.strip() == "RUN conda install python=3.8":
        continue
    if line.strip().startswith("RUN conda install pytorch==1.10.1"):
        injected.append(
            "RUN conda install -y -n bevfusion "
            "pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 "
            "-c pytorch"
        )
        continue
    if line.strip() == "RUN pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0":
        injected.append(
            "RUN pip install mmcv-full==1.4.0 "
            "-f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html"
        )
        injected.append("RUN pip install mmdet==2.20.0")
        continue
    injected.append(line)

target.write_text("\n".join(injected) + "\n", encoding="utf-8")
PY

docker build -t "${IMAGE_TAG}" -f "${TMP_DOCKERFILE}" "${BEVFUSION_ROOT}/docker"
