#!/usr/bin/env bash
set -euo pipefail

BEVFUSION_ROOT="${BEVFUSION_ROOT:-/home/achbogga/projects/bevfusion}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/storage/research/nuscenes}"
IMAGE_TAG="${IMAGE_TAG:-tsqbev-bevfusion-official:latest}"
VERSION="${VERSION:-v1.0-trainval}"
MAX_SWEEPS="${MAX_SWEEPS:-10}"
INFO_MODE="${INFO_MODE:-eval-only}"
TSQBEV_ROOT="${TSQBEV_ROOT:-/home/achbogga/projects/tsqbev-poc}"

if [[ ! -d "${BEVFUSION_ROOT}" ]]; then
  echo "missing BEVFusion repo at ${BEVFUSION_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "missing nuScenes dataset root at ${DATASET_ROOT}" >&2
  exit 1
fi

docker run --rm --gpus all --shm-size 16g \
  -v "${BEVFUSION_ROOT}:/workspace/bevfusion" \
  -v "${DATASET_ROOT}:/dataset" \
  -v "${TSQBEV_ROOT}:/workspace/tsqbev-poc" \
  -w /workspace/bevfusion \
  "${IMAGE_TAG}" \
  /bin/bash -lc "mkdir -p data && \
    ln -sfn /dataset data/nuscenes && \
    python /workspace/tsqbev-poc/research/scripts/prepare_bevfusion_nuscenes_infos.py \
      --bevfusion-root /workspace/bevfusion \
      --dataset-root /dataset \
      --version ${VERSION} \
      --mode ${INFO_MODE} \
      --max-sweeps ${MAX_SWEEPS}"
