#!/usr/bin/env bash
set -euo pipefail

BEVFUSION_ROOT="${BEVFUSION_ROOT:-/home/achbogga/projects/bevfusion}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/storage/research/nuscenes}"
IMAGE_TAG="${IMAGE_TAG:-tsqbev-bevfusion-official:latest}"
NUM_GPUS="${NUM_GPUS:-1}"
CONFIG_REL="${CONFIG_REL:-configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-pretrained/bevfusion-det.pth}"
EVAL_KIND="${EVAL_KIND:-bbox}"

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
  -w /workspace/bevfusion \
  "${IMAGE_TAG}" \
  /bin/bash -lc "mkdir -p data && \
    ln -sfn /dataset data/nuscenes && \
    python setup.py develop && \
    torchpack dist-run -np ${NUM_GPUS} python tools/test.py \
      ${CONFIG_REL} \
      ${CHECKPOINT_PATH} \
      --eval ${EVAL_KIND}"
