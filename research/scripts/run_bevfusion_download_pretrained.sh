#!/usr/bin/env bash
set -euo pipefail

BEVFUSION_ROOT="${BEVFUSION_ROOT:-/home/achbogga/projects/bevfusion}"
IMAGE_TAG="${IMAGE_TAG:-tsqbev-bevfusion-official:latest}"

if [[ ! -d "${BEVFUSION_ROOT}" ]]; then
  echo "missing BEVFusion repo at ${BEVFUSION_ROOT}" >&2
  exit 1
fi

docker run --rm --gpus all --shm-size 16g \
  -v "${BEVFUSION_ROOT}:/workspace/bevfusion" \
  -w /workspace/bevfusion \
  "${IMAGE_TAG}" \
  /bin/bash -lc "bash tools/download_pretrained.sh"
