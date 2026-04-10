#!/usr/bin/env bash
set -euo pipefail

BEVDET_ROOT="${BEVDET_ROOT:-/home/achbogga/projects/BEVDet}"
BEVDET_IMAGE_TAG="${BEVDET_IMAGE_TAG:-tsqbev-bevdet-official:latest}"

if [[ ! -f "${BEVDET_ROOT}/docker/Dockerfile" ]]; then
  echo "missing Dockerfile at ${BEVDET_ROOT}/docker/Dockerfile" >&2
  exit 1
fi

TMP_DOCKERFILE="$(mktemp /tmp/bevdet-dockerfile.XXXXXX)"
cleanup() {
  rm -f "${TMP_DOCKERFILE}"
}
trap cleanup EXIT

python3 - "${BEVDET_ROOT}/docker/Dockerfile" "${TMP_DOCKERFILE}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
target = Path(sys.argv[2])
patched = []
skip_deployment_block = False
for line in source:
    stripped = line.strip()
    if stripped == "### get onnxruntime":
        skip_deployment_block = True
        patched.append(
            "### TSQBEV patch: skip deployment-only ONNXRuntime/MMDeploy bootstrap; "
            "official BEVDet training follows the plain MMDetection3D install path."
        )
        continue
    if skip_deployment_block:
        if stripped.startswith("RUN pip install mmdet==2.25.1"):
            skip_deployment_block = False
        else:
            continue
    if stripped.startswith("RUN sed -i s:/archive.ubuntu.com:"):
        continue
    if "tuna.tsinghua.edu.cn" in line:
        line = line.replace("https://pypi.tuna.tsinghua.edu.cn/simple", "https://pypi.org/simple")
        line = line.replace(
            "-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/",
            "-c pytorch",
        )
    if "mirrors.aliyun.com" in line:
        line = line.replace("mirrors.aliyun.com", "archive.ubuntu.com")
    if "Miniconda3-latest-Linux-x86_64.sh" in line:
        line = line.replace(
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
            "https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh",
        )
    if "/opt/conda/bin/conda install -y python=${PYTHON_VERSION}" in line:
        line = line.replace(
            "/opt/conda/bin/conda install -y python=${PYTHON_VERSION}",
            "/opt/conda/bin/conda install -y",
        )
    if "/opt/conda/bin/conda install pytorch==" in line:
        line = line.replace(
            "/opt/conda/bin/conda install pytorch==",
            "/opt/conda/bin/conda install -y pytorch==",
        )
    if stripped.startswith("RUN pip install pycuda"):
        line = line.replace("pycuda \\\n", "")
        line = line.replace("pycuda ", "")
    if "networkx==2.2" in line:
        line = line.replace("numpy \\", "numpy<1.24 \\")
    patched.append(line)
target.write_text("\n".join(patched) + "\n", encoding="utf-8")
PY

docker build -t "${BEVDET_IMAGE_TAG}" -f "${TMP_DOCKERFILE}" "${BEVDET_ROOT}/docker"
