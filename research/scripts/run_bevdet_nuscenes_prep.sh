#!/usr/bin/env bash
set -euo pipefail

BEVDET_ROOT="${BEVDET_ROOT:-/home/achbogga/projects/BEVDet}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/storage/research/nuscenes}"
BEVDET_IMAGE_TAG="${BEVDET_IMAGE_TAG:-tsqbev-bevdet-official:latest}"
VERSION="${VERSION:-v1.0-mini}"
MAX_SWEEPS="${MAX_SWEEPS:-10}"
EXTRA_TAG="${EXTRA_TAG:-}"
TSQBEV_ROOT="${TSQBEV_ROOT:-/home/achbogga/projects/tsqbev-poc}"

if [[ -z "${EXTRA_TAG}" ]]; then
  if [[ "${VERSION}" == "v1.0-mini" ]]; then
    EXTRA_TAG="bevdetv3-mini"
  else
    EXTRA_TAG="bevdetv3-nuscenes"
  fi
fi

if docker image inspect "${BEVDET_IMAGE_TAG}" >/dev/null 2>&1; then
  docker run --rm --shm-size 8g \
    -v "${BEVDET_ROOT}:/workspace/BEVDet" \
    -v "${DATASET_ROOT}:/dataset" \
    -v "${TSQBEV_ROOT}:/workspace/tsqbev-poc" \
    -w /workspace/BEVDet \
    "${BEVDET_IMAGE_TAG}" \
    /bin/bash -lc "python /workspace/tsqbev-poc/research/scripts/prepare_bevdet_nuscenes_infos.py \
      --bevdet-root /workspace/BEVDet \
      --dataset-root /dataset \
      --version '${VERSION}' \
      --extra-tag '${EXTRA_TAG}' \
      --max-sweeps '${MAX_SWEEPS}'"
else
  uv run --project "${TSQBEV_ROOT}" python "${TSQBEV_ROOT}/research/scripts/prepare_bevdet_nuscenes_infos.py" \
    --bevdet-root "${BEVDET_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --version "${VERSION}" \
    --extra-tag "${EXTRA_TAG}" \
    --max-sweeps "${MAX_SWEEPS}"
fi
