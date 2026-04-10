#!/usr/bin/env bash
set -euo pipefail

BEVDET_ROOT="${BEVDET_ROOT:-/home/achbogga/projects/BEVDet}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/storage/research/nuscenes}"
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

uv run python "${TSQBEV_ROOT}/research/scripts/prepare_bevdet_nuscenes_infos.py" \
  --bevdet-root "${BEVDET_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --version "${VERSION}" \
  --extra-tag "${EXTRA_TAG}" \
  --max-sweeps "${MAX_SWEEPS}"
